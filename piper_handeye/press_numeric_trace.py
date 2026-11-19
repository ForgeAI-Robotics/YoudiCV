"""Preallocated numeric per-decision capture; serialization only after motion."""
from dataclasses import asdict
import json
import hashlib
from pathlib import Path

import numpy as np


def observer_fingerprint():
    # The OrangePi and AGX UDP client wrappers differ; their pure observer
    # implementations are identical. Fingerprint that exact source prefix,
    # rather than incorrectly rejecting a capture because the client differs.
    from . import guarded_press
    source=Path(guarded_press.__file__).read_text()
    prefix=source.split('\ndef generate_bounded_retract_trajectory(',1)[0]
    if 'class GuardedPressMonitor:' not in prefix:
        raise ValueError('pure contact observer source could not be identified')
    return hashlib.sha256(prefix.encode()).hexdigest()


def candidate_fingerprint():
    from . import contact_prevention
    return hashlib.sha256(Path(contact_prevention.__file__).read_bytes()).hexdigest()


def candidate_options(monitor):
    return dict(continuous_evidence=monitor.continuous_evidence,
        measured_progress=monitor.measured_progress,
        motion_reference=monitor.motion_reference_enabled,
        encoder_max_age_sec=monitor.encoder_max_age_sec)


def validate_capture_source(metadata):
    if metadata.get('producer') not in ('original', 'candidate'):
        raise ValueError('unknown capture observer')
    if metadata.get('original_observer_sha256') != observer_fingerprint():
        raise ValueError('numeric capture observer source differs; exact replay is unavailable')
    if metadata['producer'] == 'candidate':
        if metadata.get('candidate_observer_sha256') != candidate_fingerprint():
            raise ValueError('numeric capture candidate source differs; exact replay is unavailable')
        options = metadata.get('candidate_observer_options')
        required = {'continuous_evidence', 'measured_progress', 'motion_reference',
                    'encoder_max_age_sec'}
        if not isinstance(options, dict) or set(options) != required:
            raise ValueError('numeric capture candidate settings are missing')


class NumericPressTrace:
    # Inputs sufficient to replay the pure observer from its original baseline.
    columns = [('sample_time',1),('decision_time',1),('elapsed',1),('planned_travel',1),
        ('encoder_age',1),('raw_current',6),('measured_joints',6),('desired_joints',6),
        ('filtered_current',6),('free_current',6),('free_sigma',6),
        ('decision',1),('normal_hold',1),('normal_force',1),('normal_sigma',1),
        ('normal_score',1),('normal_alignment',1),('tangential_force',1),
        ('tangential_score',1),('total_force',1)]

    def __init__(self, capacity=2000):
        if capacity<1:raise ValueError('trace capacity must be positive')
        self.slices={};offset=0
        for name,width in self.columns:
            self.slices[name]=slice(offset,offset+width);offset+=width
        self.data=np.empty((capacity,offset),dtype=np.float64)
        self.used=0;self.dropped=0;self.metadata=None

    def configure(self, baseline, config, normal, producer, monitor=None):
        if producer not in ('original', 'candidate'):
            raise ValueError('unknown capture observer')
        if producer == 'candidate' and monitor is None:
            raise ValueError('candidate capture requires the actual observer settings')
        cfg=asdict(config)
        cfg={k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in cfg.items()}
        self.metadata=dict(schema=2,producer=producer,full_decision_capture=True,
            original_observer_sha256=observer_fingerprint(),
            fingerprint_scope='pure observer prefix; excludes UDP client wrappers',
            baseline_current_a=baseline.median_a.tolist(),baseline_sigma_a=baseline.sigma_a.tolist(),
            baseline_samples=baseline.sample_count,current_guard_config=cfg,
            contact_normal_base=np.asarray(normal).tolist())
        if producer == 'candidate':
            self.metadata.update(candidate_observer_sha256=candidate_fingerprint(),
                candidate_observer_options=candidate_options(monitor))

    def record(self,sample,now,elapsed,planned,measured,age,desired,monitor,result):
        if self.used>=len(self.data):self.dropped+=1;return
        row=self.data[self.used];s=self.slices
        row[s['sample_time']]=sample.timestamp;row[s['decision_time']]=now
        row[s['elapsed']]=elapsed;row[s['planned_travel']]=planned
        row[s['encoder_age']]=np.nan if age is None else age
        row[s['raw_current']]=sample.currents_a
        row[s['measured_joints']]=np.nan if measured is None else measured
        row[s['desired_joints']]=desired;row[s['filtered_current']]=result.filtered_current_a
        row[s['free_current']]=monitor.relative_reference_a;row[s['free_sigma']]=monitor.relative_sigma_a
        row[s['decision']]=0 if result.decision.value=='CONTINUE' else (1 if result.decision.value=='CONTACT' else 2)
        row[s['normal_hold']]=result.persistence_sec
        for key,attr in [('normal_force','estimated_normal_force_n'),('normal_sigma','normal_force_sigma_n'),
            ('normal_score','normal_force_score'),('normal_alignment','normal_force_alignment'),
            ('tangential_force','estimated_tangential_force_n'),('tangential_score','tangential_force_score'),
            ('total_force','estimated_total_force_n')]:
            value=getattr(result,attr);row[s[key]]=np.nan if value is None else value
        self.used+=1

    def save(self,path,**context):
        if self.metadata is None:return None
        path=Path(path);temp=path.with_name(path.name+'.tmp')
        meta=dict(self.metadata,**context,samples=self.used,dropped=self.dropped,
                  complete=self.dropped==0,columns=self.columns)
        with temp.open('wb') as f:
            np.savez(f,data=self.data[:self.used],metadata=np.asarray(json.dumps(meta)))
        temp.replace(path)
        return dict(path=str(path),samples=self.used,dropped=self.dropped,complete=meta['complete'])


def load_numeric_trace(path):
    with np.load(path,allow_pickle=False) as archive:
        data=archive['data'];meta=json.loads(str(archive['metadata']))
    if not meta.get('complete') or meta.get('dropped'):
        raise ValueError('incomplete numeric capture cannot be used as an exact replay')
    if len(data) == 0 or len(data) != meta.get('samples'):
        raise ValueError('numeric capture sample count is empty or inconsistent')
    slices={};offset=0
    for name,width in meta['columns']:slices[name]=slice(offset,offset+width);offset+=width
    def scalar(row,name):return float(row[slices[name]][0])
    rows=[]
    for row in data:
        measured=row[slices['measured_joints']]
        rows.append(dict(sample_timestamp_monotonic=scalar(row,'sample_time'),
            processing_monotonic_sec=scalar(row,'decision_time'),elapsed_sec=scalar(row,'elapsed'),
            forward_travel_m=scalar(row,'planned_travel'),raw_current_a=row[slices['raw_current']].tolist(),
            measured_joints_rad=measured.tolist() if np.isfinite(measured).all() else None,
            desired_joints_rad=row[slices['desired_joints']].tolist(),
            measured_joint_age_sec=scalar(row,'encoder_age'),
            recorded_decision=['CONTINUE','CONTACT','HARD_STOP'][int(scalar(row,'decision'))]))
    return meta,rows
