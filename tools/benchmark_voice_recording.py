"""Benchmark recorded audio only. No microphone, robot or command dispatcher."""
import argparse
import json
from pathlib import Path
import sys
import time
import wave
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from piper_handeye.voice_audio import AdaptiveEnergyVad, WhisperServer
from piper_handeye.voice_command import OneShotConsensusSession

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--input', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--repeats', type=int, default=3)
a = p.parse_args()
with wave.open(str(a.input)) as w:
    if (w.getnchannels(), w.getsampwidth(), w.getframerate()) != (1,2,16000):
        raise ValueError('require 16kHz mono PCM16 WAV')
    pcm = w.readframes(w.getnframes())
vad = AdaptiveEnergyVad()
segments = []
for offset in range(0, len(pcm)-vad.frame_bytes+1, vad.frame_bytes):
    utterance = vad.feed(pcm[offset:offset+vad.frame_bytes])
    if utterance is not None:
        segments.append((f'vad_{len(segments)+1}', utterance.pcm))
inputs = [('whole_recording', pcm), *segments]
report = {'input': str(a.input.resolve()), 'recording_sec':len(pcm)/32000,
          'vad_segments':len(segments), 'model':'ggml-small.bin',
          'gpu_requested':True, 'threads':4, 'rows':[],
          'note':'No real-time playback; VAD replay wall time excludes recording and endpoint wait.'}
server = WhisperServer(ROOT/'third_party/whisper.cpp/build/bin/whisper-server',
                       ROOT/'third_party/whisper.cpp/models/ggml-small.bin', port=18178)
original = server._transcribe_path
calls=[]
def timed(*args, **kwargs):
    start=time.perf_counter()
    result=original(*args, **kwargs)
    calls.append(time.perf_counter()-start)
    return result
server._transcribe_path=timed
a.output.parent.mkdir(parents=True,exist_ok=True)
try:
    start=time.perf_counter();server.start();report['startup_sec']=time.perf_counter()-start
    print(json.dumps({'startup_sec':report['startup_sec'],'vad_segments':len(segments)}),flush=True)
    for index in range(a.repeats):
        for name, audio in inputs:
            calls.clear(); start=time.perf_counter()
            primary,verifier=server.transcribe_consensus(audio)
            elapsed=time.perf_counter()-start
            decision=OneShotConsensusSession().submit(*primary,*verifier)
            row=dict(run=index+1,input_kind=name,audio_sec=len(audio)/32000,
                     total_sec=elapsed,primary_sec=calls[0],verifier_sec=calls[1],
                     preprocessing_and_overhead_sec=elapsed-sum(calls),
                     primary_text=primary[0],primary_confidence=primary[1],
                     verifier_text=verifier[0],verifier_confidence=verifier[1],
                     decision=decision.status.value,reason=decision.message,
                     command=None if decision.command is None else decision.command.signature)
            report['rows'].append(row);a.output.write_text(json.dumps(report,ensure_ascii=False,indent=2))
            print(json.dumps(row,ensure_ascii=False),flush=True)
finally:
    if server._stderr_file is not None:
        server._stderr_file.flush();server._stderr_file.seek(0)
        a.output.with_suffix('.whisper.log').write_bytes(server._stderr_file.read())
    server.close()
    a.output.write_text(json.dumps(report,ensure_ascii=False,indent=2))
