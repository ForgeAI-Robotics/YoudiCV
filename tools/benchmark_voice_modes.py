"""Compare single and consensus ASR on saved recordings; never dispatch commands."""
import argparse
import importlib.util
import json
from pathlib import Path
import statistics
import sys
import time
import wave
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from piper_handeye.voice_audio import AdaptiveEnergyVad,WhisperServer,lowpass_pcm16
from piper_handeye.voice_command import OneShotConsensusSession,parse_voice_command
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--model',choices=['tiny','base','small'],default='small')
parser.add_argument('--output',type=Path,required=True)
parser.add_argument('--repeats',type=int,default=2)
args=parser.parse_args()
out=args.output;out.parent.mkdir(parents=True,exist_ok=True)
inputs=[];filters=[]
spec=importlib.util.spec_from_file_location('old_voice_audio',ROOT.parent/'snapshots/20260908_110636_button15_near_depth_pause/source/contral_test/piper_handeye/voice_audio.py');old=importlib.util.module_from_spec(spec);sys.modules[spec.name]=old;spec.loader.exec_module(old)
lowpass_pcm16(bytes(640))
for path in sorted((ROOT/'voice_diagnostics').glob('*/mono_16k.wav')):
    with wave.open(str(path)) as w: pcm=w.readframes(w.getnframes())
    before=[];after=[]
    for _ in range(3):
        t=time.perf_counter();a=old.lowpass_pcm16(pcm);before.append(time.perf_counter()-t)
        t=time.perf_counter();b=lowpass_pcm16(pcm);after.append(time.perf_counter()-t)
    import numpy as np
    delta=np.abs(np.frombuffer(a,dtype='<i2').astype(int)-np.frombuffer(b,dtype='<i2').astype(int))
    filters.append(dict(input=str(path.relative_to(ROOT)),before_sec=statistics.median(before),after_sec=statistics.median(after),max_pcm_delta=int(delta.max())))
    vad=AdaptiveEnergyVad();count=0
    for offset in range(0,len(pcm)-vad.frame_bytes+1,vad.frame_bytes):
        u=vad.feed(pcm[offset:offset+vad.frame_bytes])
        if u is not None:
            count+=1;inputs.append((f'{path.parent.name}/vad_{count}',u.pcm))
    if not count: inputs.append((f'{path.parent.name}/whole_no_vad',pcm))
inputs.append(('synthetic_silence_2sec',bytes(64000)))
report={'model':args.model,'filters':filters,'rows':[],'physical_motion_executed':False,'note':'Repeated saved recordings, no ground-truth accuracy claim. Silence is synthetic negative control.'}
print(json.dumps({'filters':filters,'inputs':[(n,len(p)/32000) for n,p in inputs]}),flush=True)
server=WhisperServer(ROOT/'third_party/whisper.cpp/build/bin/whisper-server',ROOT/f'third_party/whisper.cpp/models/ggml-{args.model}.bin',port=18178)
try:
    t=time.perf_counter();server.start();report['startup_sec']=time.perf_counter()-t
    for repeat in range(args.repeats):
        for name,pcm in inputs:
            t=time.perf_counter();single=server.transcribe(pcm);single_sec=time.perf_counter()-t
            t=time.perf_counter();primary,verifier=server.transcribe_consensus(pcm);double_sec=time.perf_counter()-t
            command=parse_voice_command(single[0]);decision=OneShotConsensusSession().submit(*primary,*verifier)
            row=dict(run=repeat+1,input=name,audio_sec=len(pcm)/32000,single_sec=single_sec,consensus_sec=double_sec,single=single,primary=primary,verifier=verifier,single_command=command.signature,consensus_status=decision.status.value,consensus_command=None if decision.command is None else decision.command.signature)
            report['rows'].append(row);out.write_text(json.dumps(report,ensure_ascii=False,indent=2));print(json.dumps(row,ensure_ascii=False),flush=True)
finally:
    if server._stderr_file:
        server._stderr_file.flush();server._stderr_file.seek(0);out.with_suffix('.whisper.log').write_bytes(server._stderr_file.read())
    server.close();out.write_text(json.dumps(report,ensure_ascii=False,indent=2))
