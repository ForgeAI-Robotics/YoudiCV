#!/usr/bin/env python3
"""Pure offline current replay. Never imports ROS or creates a network socket.

Historical traces are decimated, and end at the original stop. All results are
algorithm diagnostics, not physical press success rates or full-rate replays.
"""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from piper_handeye.guarded_press import (GuardedPressConfig, GuardedPressMonitor,
    MotorCurrentSample, RobustCurrentBaseline)
from piper_handeye.contact_prevention import CandidateGuardedPressMonitor
from piper_handeye.press_numeric_trace import (load_numeric_trace,
    validate_capture_source, candidate_options)


def make_config(source):
    source=dict(source)
    return GuardedPressConfig(**source)


def replay(sample, trace, variant):
    if sample.get('full_decision_capture'):
        validate_capture_source(sample)
    config=make_config(sample['current_guard_config'])
    base=RobustCurrentBaseline(np.asarray(sample['baseline_current_a']),
                              np.asarray(sample['baseline_sigma_a']),sample.get('baseline_samples',40))
    original=variant in ('original_sparse_input','original_full_rate')
    if original:
        monitor=GuardedPressMonitor(base,config,contact_normal_base=sample['contact_normal_base'])
    else:
        options=dict(continuous_evidence=variant in ('candidate_strict_continuity', 'candidate_full_rate'),
            measured_progress=variant in ('measured_progress_sparse_diagnostic',
                'pose_reference_sparse_diagnostic','candidate_strict_continuity','candidate_full_rate'),
            motion_reference=variant in ('pose_reference_sparse_diagnostic',
                'candidate_strict_continuity','candidate_full_rate'))
        if variant == 'candidate_full_rate':
            if not sample.get('full_decision_capture') or sample.get('producer') != 'candidate':
                raise ValueError('candidate_full_rate requires a complete candidate capture')
            options=sample['candidate_observer_options']
        monitor=CandidateGuardedPressMonitor(base,config,
            contact_normal_base=sample['contact_normal_base'], **options)
    same_observer = (sample.get('producer') == 'original' if original else
        sample.get('producer') == 'candidate' and
        sample.get('candidate_observer_options') == candidate_options(monitor))
    result=None
    stop=None
    model_status=Counter()
    checkpoint_matches=[]
    for row in trace:
        # Processing age was not logged. Do not invent historic latency:
        # explicitly use a zero-age sample clock, while retaining real gaps.
        stamp=float(row['sample_timestamp_monotonic'])
        kwargs=dict(now=row.get('processing_monotonic_sec',stamp),elapsed_sec=row['elapsed_sec'],
            forward_travel_m=row['forward_travel_m'],joints_rad=row.get('measured_joints_rad'))
        if original:
            if kwargs['joints_rad'] is None:
                kwargs['joints_rad']=row.get('desired_joints_rad')
        else:
            kwargs['encoder_age_sec']=row.get('measured_joint_age_sec')
        result=monitor.update(MotorCurrentSample(stamp,np.asarray(row['raw_current_a'])),**kwargs)
        if 'recorded_decision' in row:
            checkpoint_matches.append(result.decision.value==row['recorded_decision'])
        if isinstance(monitor,CandidateGuardedPressMonitor):
            status=monitor.candidate_audit.get('motion_reference',{}).get('status')
            if status:model_status[status]+=1
        if result.decision.value!='CONTINUE':
            stop=row
            break
    stopped=stop is not None
    return dict(variant=variant,decision=result.decision.value if result else 'NO_INPUT',
        reason=result.reason if result else 'empty_trace',
        stop_at_recorded_progress_mm=1000*stop['forward_travel_m'] if stopped else None,
        original_stop_progress_mm=sample['travel_mm'],
        stop_difference_mm=1000*stop['forward_travel_m']-sample['travel_mm'] if stopped else None,
        endpoint_censored=not stopped,
        interpretation=('changed_or_reproduced_decision_within_saved_inputs' if stopped
                        else 'no_decision_before_original_stop; subsequent motion and outcome unknown'),
        continuity_resets=getattr(monitor,'continuity_resets',0),
        matching_captured_observer=bool(same_observer),
        captured_decision_sequence_matches=(all(checkpoint_matches) and len(checkpoint_matches)==len(trace))
            if checkpoint_matches and same_observer else None,
        motion_reference_status_counts=dict(model_status))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    source=parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--samples',type=Path)
    source.add_argument('--bundle',type=Path,help='Complete numeric capture, with its baseline and settings')
    parser.add_argument('--traces',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--limit',type=int)
    parser.add_argument('--variants',nargs='+',default=[
        'original_sparse_input','measured_progress_sparse_diagnostic',
        'pose_reference_sparse_diagnostic','candidate_strict_continuity'],
        choices=['original_sparse_input','original_full_rate','measured_progress_sparse_diagnostic',
                 'pose_reference_sparse_diagnostic','candidate_strict_continuity','candidate_full_rate'])
    args=parser.parse_args()
    # An accidental future use of a robot client must fail, including localhost.
    import socket
    def no_network(*a,**kw):
        raise RuntimeError('network access prohibited in offline press replay')
    socket.socket=no_network
    bundle_rows=None
    if args.bundle is not None:
        meta,bundle_rows=load_numeric_trace(args.bundle)
        validate_capture_source(meta)
        samples=[dict(meta,round=0,button='not_independently_labeled',attempt=1,
            bridge_task_id=meta.get('task_id','numeric_capture'),state='UNKNOWN',
            reason=meta.get('historical_reason'),trace=str(args.bundle),
            travel_mm=1000*bundle_rows[-1]['forward_travel_m'])]
    else:
        if args.traces is None:parser.error('--traces is required with --samples')
        samples=json.loads(args.samples.read_text())
    if args.limit is not None:samples=samples[:args.limit]
    args.output.mkdir(parents=True,exist_ok=True)
    results=[];gap_stats=[]
    start=time.monotonic()
    with (args.output/'replay_results.jsonl').open('w') as log:
        for index,sample in enumerate(samples):
            if bundle_rows is not None:trace=bundle_rows
            else:
                path=args.traces/Path(sample['trace']).name
                trace=[json.loads(line) for line in path.read_text().splitlines() if line.strip()]
            stamps=np.asarray([r['sample_timestamp_monotonic'] for r in trace])
            gaps=np.diff(stamps)
            gap_stats.append(dict(bridge_task_id=sample['bridge_task_id'],rows=len(trace),
                median_saved_gap_sec=float(np.median(gaps)),maximum_saved_gap_sec=float(np.max(gaps)),
                saved_rows_exceed_30ms=bool(np.any(gaps>.03))))
            for variant in args.variants:
                r=replay(sample,trace,variant)
                r.update(round=sample['round'],button=sample['button'],attempt=sample['attempt'],
                    bridge_task_id=sample['bridge_task_id'],observed_post_press_state=sample['state'],
                    historically_recorded_reason=sample['reason'],historical_trace_rows=len(trace))
                results.append(r);log.write(json.dumps(r,ensure_ascii=False)+'\n')
            if (index+1)%100==0:
                print(json.dumps(dict(processed=index+1,total=len(samples),elapsed_sec=round(time.monotonic()-start,2))),flush=True)
    summary=dict(press_samples=len(samples),replay_runs=len(results),physical_motion_sent=False,
        network_prohibited=True,production_source_modified=False,
        full_rate_numeric_input=bundle_rows is not None,
        input_policy=('all captured observer inputs, actual decision timestamps, no interpolation' if bundle_rows is not None
            else 'original retained rows and timestamps, no interpolation, no resampling, zero processing-age assumption'),
        limitations=[
            'Input logging omitted about nine of every ten online decisions; this is not an exact production replay.',
            'Intervals between saved rows are logging gaps, not proof of CAN sensor loss.',
            'Strict continuity is expected to reject/clear sparse inputs. It cannot be graded for physical performance here.',
            'Traces end at the original stop. A candidate continuing at that point has an unknown later outcome.',
            'ON/OFF labels are program observations, not independently measured first contact or button travel.',
            'Candidate current model is a bounded local pose predictor, not an identified dynamics/friction model.'],
        variants={},sampling=dict(traces_with_saved_gap_over_30ms=sum(x['saved_rows_exceed_30ms'] for x in gap_stats),
            median_of_saved_median_gaps_sec=float(np.median([x['median_saved_gap_sec'] for x in gap_stats]))),
        elapsed_sec=time.monotonic()-start)
    if bundle_rows is not None:
        summary['limitations']=[
            'Full-rate capture preserves all observer calls; it does not measure physical first contact or button plunger travel.',
            'The trace ends at the original stop. Continuing past it has an unknown later physical outcome.',
            'Candidate current model is a bounded local pose predictor, not an identified dynamics/friction model.']
    for variant in args.variants:
        rows=[r for r in results if r['variant']==variant]
        summary['variants'][variant]=dict(decisions=dict(Counter(r['decision'] for r in rows)),
            reasons=dict(Counter(r['reason'] for r in rows)),endpoint_censored=sum(r['endpoint_censored'] for r in rows),
            same_reason_within_2mm_of_recorded_stop=sum(r['reason']==r['historically_recorded_reason']
                and r['stop_difference_mm'] is not None and abs(r['stop_difference_mm'])<=2. for r in rows),
            by_observed_state={state:dict(decisions=dict(Counter(r['decision'] for r in rows
                if r['observed_post_press_state']==state)),censored=sum(r['endpoint_censored'] for r in rows
                if r['observed_post_press_state']==state)) for state in ('ON','OFF','UNKNOWN')})
    (args.output/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    (args.output/'sampling_audit.json').write_text(json.dumps(gap_stats,ensure_ascii=False,indent=2)+'\n')
    flat=[{k:v for k,v in r.items() if k!='motion_reference_status_counts'} for r in results]
    if flat:
        with (args.output/'replay_results.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
    print(json.dumps(summary,ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
