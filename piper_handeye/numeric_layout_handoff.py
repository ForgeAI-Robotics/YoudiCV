"""Explicit, opt-in layout provenance at existing task boundaries.

Inferred ROIs and poses stay separate from YOLO detections and have no target
confidence. This module never issues a robot command or changes a target.
"""
from copy import deepcopy
from dataclasses import dataclass
import time

import numpy as np

from .numeric_layout_sampling import LayoutTransform, _LayoutWindow, rigid_transform
from .realtime_planner import SlidingButtonWindow, TimedButtonTransform, WindowEstimate
from .numeric_yolo_consistency import target_conflicted


class LayoutTaskPolicy:
    def __init__(self, settings, target):
        self.settings = dict(settings or {})
        self.target = str(target)
        self.enabled = (self.settings.get('task_fallback_enabled') is True
                        and self.settings.get('mode') == 'shadow'
                        and self.target in ('1','2','3','4','5')
                        and self.target in self.settings.get('target_labels', []))
        self.digest = self.settings.get('profile_sha256')
        if self.enabled and (not isinstance(self.digest,str) or len(self.digest)!=64
                             or any(c not in '0123456789abcdef' for c in self.digest)):
            raise ValueError('invalid task layout profile SHA256')

    def evidence(self, record, *, sequence=None, capture=None, now=None, depth=False, max_frame_age_sec=.35):
        if not self.enabled or not isinstance(record,dict):
            return None
        try:
            now = time.time() if now is None else float(now)
            # Only explicit 2D lamp consumers may request a longer stationary age.
            age_limit = .35 if depth else float(max_frame_age_sec)
            if not .35 <= age_limit <= .7:return None
            seq = record['frame_sequence']; stamp = float(record['capture_time_unix_sec'])
            anchors = record['geometry']['anchors']
            box = np.asarray(record['bbox_xyxy'],float)
            width,height = map(float,record['image_size'])
            valid = (type(seq) is int and seq > 0 and record['rgbd_bundle_sequence']==seq
                     and (sequence is None or sequence==seq)
                     and (capture is None or float(capture)==stamp)
                     and -.05 <= now-stamp <= age_limit
                     and record['source']=='panel_layout_inferred' and record['mode']=='shadow'
                     and record['target_label']==self.target and record['target_detected'] is False
                     and record['yolo_confidence'] is None and record['motion_eligible'] is False
                     and record['profile_sha256']==self.digest and bool(record['profile_id'])
                     and record['pairing']=='same_rgbd_bundle' and record['accepted_2d'] is True
                     and record['geometry']['accepted'] is True
                     and len(anchors)>=3 and len(anchors)==len(set(anchors)) and self.target not in anchors
                     and all(str(a) in tuple(map(str,range(1,17))) for a in anchors)
                     and box.shape==(4,) and np.isfinite(box).all()
                     and np.isfinite([width,height]).all()
                     and 5<=box[0]<box[2]<=width-5 and 5<=box[1]<box[3]<=height-5
                     and min(box[2]-box[0],box[3]-box[1])>=14)
            if not valid:return None
            if depth:
                pose=record['pose']
                matrix=rigid_transform(pose['camera_to_button'],'layout camera_to_button')
                point=np.asarray(pose['p_camera_m'],float)
                normal=np.asarray(pose['normal_camera'],float)
                if (record['valid_3d'] is not True or int(pose['panel_support_detections'])<3
                        or not 0<=float(pose['plane_rms_m'])<=.006
                        or point.shape!=(3,) or normal.shape!=(3,)
                        or not np.isfinite(point).all() or not np.isfinite(normal).all()
                        or not np.allclose(point,matrix[:3,3],atol=1e-6,rtol=0)
                        or abs(np.linalg.norm(normal)-1)>.001
                        or abs(normal@matrix[:3,2])<.999):return None
            return deepcopy(record)
        except (KeyError,TypeError,ValueError,OverflowError):
            return None

    def region(self, record, *, sequence=None, capture=None, now=None, max_frame_age_sec=.35):
        evidence=self.evidence(record,sequence=sequence,capture=capture,now=now,
                               max_frame_age_sec=max_frame_age_sec)
        if evidence is None:return None
        return dict(label=self.target,source='panel_layout_inferred',target_detected=False,
                    confidence=None,frame_sequence=evidence['frame_sequence'],
                    bbox_xyxy=list(evidence['bbox_xyxy']),profile_sha256=self.digest,
                    layout_evidence=evidence,layout_policy=deepcopy(self.settings))

    def sample(self, record, *, now=None):
        e=self.evidence(record,now=now,depth=True)
        if e is None:return None
        return dict(sequence=e['frame_sequence'],target_label=self.target,
                    capture_time_unix_sec=e['capture_time_unix_sec'],received_monotonic=e['received_monotonic'],
                    camera_to_button=e['pose']['camera_to_button'],plane_rms_m=e['pose']['plane_rms_m'],
                    bbox_xyxy=e['bbox_xyxy'],confidence=None,position_source='panel_layout_inferred',
                    target_detected=False,layout_evidence=e,layout_policy=deepcopy(self.settings))


def selected_button_region(snapshot, *, now=None, target_label=None, max_frame_age_sec=.35):
    """Ordinary YOLO region wins. Only an explicitly admitted layout ROI follows."""
    detection=snapshot.get('target_detection')
    if detection is not None:return detection
    region=snapshot.get('layout_target_region')
    if region is None and max_frame_age_sec > .35:
        region=snapshot.get('stationary_light_layout_candidate')
    if not isinstance(region,dict):return None
    expected_target=target_label or snapshot.get('layout_requested_target_label')
    if expected_target is not None and str(region.get('label'))!=str(expected_target):return None
    policy=LayoutTaskPolicy(region.get('layout_policy'),region.get('label'))
    expected=policy.region(region.get('layout_evidence'),sequence=snapshot.get('frame_sequence'),
                           capture=snapshot.get('capture_time_unix_sec'),now=now,
                           max_frame_age_sec=max_frame_age_sec)
    if (expected is None or region.get('bbox_xyxy')!=expected['bbox_xyxy']
            or region.get('frame_sequence')!=expected['frame_sequence']
            or snapshot.get('image_size')!=expected['layout_evidence']['image_size']
            or region.get('confidence','missing') is not None
            or region.get('target_detected') is not False):return None
    return expected


def region_is_inferred(region):
    return isinstance(region,dict) and region.get('source')=='panel_layout_inferred'


class LayoutTaskVision:
    """Decorate existing responses; never append guessed labels to detections."""
    def __init__(self,client,policy):
        self.client=client;self.layout_task_policy=policy

    def __getattr__(self,name):return getattr(self.client,name)

    def request(self,command,**options):
        response=self.client.request(command,**options)
        policy=self.layout_task_policy
        target=str(options.get('target_label',policy.target))
        if target!=policy.target:return response
        record=response.get('numeric_layout_localization')
        audit=response.get('numeric_yolo_consistency')
        if (policy.settings.get('yolo_consistency_enabled') is True
                and command=='visible_buttons'
                and target_conflicted(audit,target=policy.target,
                    sequence=response.get('frame_sequence'),digest=policy.digest)):
            raw=response.get('detections',[])
            response={**response,'yolo_detections':raw,
                'detections':[d for d in raw if str(d.get('label'))!=policy.target]}
        if command in {'light_snapshot','visible_buttons'}:
            region=policy.region(record,sequence=response.get('frame_sequence'),
                                 capture=response.get('capture_time_unix_sec'))
            if region is not None:
                response={**response,'layout_target_region':region,'layout_requested_target_label':policy.target}
            elif command=='light_snapshot':
                candidate=policy.region(record,sequence=response.get('frame_sequence'),
                    capture=response.get('capture_time_unix_sec'),max_frame_age_sec=.7)
                if candidate is not None:
                    # Candidate is never selected by ordinary/motion callers.
                    response={**response,'stationary_light_layout_candidate':candidate,
                              'layout_requested_target_label':policy.target}
        elif command=='latest':
            sample=policy.sample(record)
            if sample is not None:response={**response,'layout_sample':sample}
        return response


def task_vision(client, config, target):
    policy=LayoutTaskPolicy(config.get('numeric_layout_localization'),target)
    return LayoutTaskVision(client,policy) if policy.enabled else client


def require_layout_server_support(health,config,target):
    policy=LayoutTaskPolicy(config.get('numeric_layout_localization'),target)
    if policy.enabled:
        support=health.get('numeric_layout_task_handoff') or {}
        if (support.get('version')!=1 or support.get('enabled') is not True
                or support.get('profile_sha256')!=policy.digest):
            raise RuntimeError('layout fallback requires matching vision/safety task handoff v1')
        if policy.settings.get('yolo_consistency_enabled') is True:
            check=health.get('numeric_yolo_consistency') or {}
            if (check.get('version')!=1 or check.get('enabled') is not True
                    or check.get('profile_sha256')!=policy.digest):
                raise RuntimeError('numeric YOLO consistency requires matching vision capability v1')


@dataclass(frozen=True)
class SourcedWindowEstimate(WindowEstimate):
    position_sources: tuple[str,...] = ()


class SourceAwareButtonWindow(_LayoutWindow):
    """Original YOLO admission; independently validated layout samples have no confidence."""
    @classmethod
    def from_window(cls,window):
        return cls(**{name:getattr(window,name) for name in (
            'size','min_confidence','max_input_age_sec','max_latest_age_sec','max_span_sec',
            'max_plane_rms_m','max_position_rms_m','max_normal_rms_deg','max_position_jump_m')})

    def add(self,sample,*,now=None):
        if isinstance(sample,TimedButtonTransform):
            return SlidingButtonWindow.add(self,sample,now=now)
        if isinstance(sample,LayoutTransform):return super().add(sample,now=now)
        raise TypeError('unsupported target sample type')

    def estimate(self,*,now=None):
        estimate=super().estimate(now=now)
        sources=tuple(sorted({'yolo' if isinstance(s,TimedButtonTransform) else 'panel_layout_inferred'
                              for s in self._samples}))
        return SourcedWindowEstimate(**vars(estimate),position_sources=sources)


def localization_provenance(estimate,audit):
    sources=getattr(estimate,'position_sources',())
    if not sources and isinstance(audit,list):
        sources=tuple(sorted({r.get('source','yolo') for r in audit if r.get('accepted')}))
    sources=sources or ('yolo',)
    return dict(position_sources=list(sources),
                target_detected_in_current_frame='panel_layout_inferred' not in sources)


def timed_target(sample,base_to_button,*,target_label,now=None):
    if sample.get('position_source')=='panel_layout_inferred':
        policy=LayoutTaskPolicy(sample.get('layout_policy'),target_label)
        expected=policy.sample(sample.get('layout_evidence'),now=now)
        if expected is None or any(sample.get(k)!=expected[k] for k in (
                'sequence','capture_time_unix_sec','camera_to_button','plane_rms_m','confidence')):
            raise ValueError('invalid inferred execution sample')
        return LayoutTransform(int(sample['sequence']),float(sample['capture_time_unix_sec']),
                               rigid_transform(base_to_button,'base_to_button'),float(sample['plane_rms_m']))
    return TimedButtonTransform(int(sample['sequence']),float(sample['capture_time_unix_sec']),
                                base_to_button,float(sample['confidence']),float(sample['plane_rms_m']))


def target_batch(response, *, after_sequence=0):
    ordinary=response.get('samples')
    if ordinary is None:ordinary=[response['sample']] if response.get('sample') else []
    samples=list(ordinary)
    layout=response.get('layout_sample')
    if layout is not None and not any(int(s['sequence'])==int(layout['sequence']) for s in samples):
        samples.append(layout)
    return sorted((s for s in samples if int(s['sequence'])>after_sequence),key=lambda s:int(s['sequence']))


def collect_stationary_task_target(vision,robot,handeye,target_label,config,**options):
    from .numeric_layout_sampling import collect_stationary_layout_shadow
    if not isinstance(vision,LayoutTaskVision) or not vision.layout_task_policy.enabled:
        raise ValueError('explicit task layout policy is required')
    try:
        result=collect_stationary_layout_shadow(vision,robot,handeye,target_label,config,
                                               _task_handoff=True,**options)
    except RuntimeError as exc:
        if hasattr(exc,'layout_sampling'):
            failure=RuntimeError('no stable fresh button target before motion')
            failure.stationary_target_failure=exc.layout_sampling
            raise failure from exc
        raise
    sources=tuple(sorted({r['source'] for r in result['audit'] if r.get('accepted')}))
    estimate=SourcedWindowEstimate(result['stable'],tuple(result['reasons']),
        np.asarray(result['base_to_button'],float),result['sequence_range'],result['sample_count'],
        result['latest_age_sec'],result['span_sec'],result['position_rms_m'],result['normal_rms_deg'],
        result['max_position_jump_m'],sources)
    audit=result['audit']
    for row in audit:
        if row.get('source')=='panel_layout_inferred':
            row['target_detected']=False;row['confidence']=None
    if audit:
        audit[-1]['acquisition_summary']=dict(target_label=str(target_label),status='ready',
            position_sources=sorted({r['source'] for r in audit if r.get('accepted')}),
            frame_count=result['frame_count'],profile_sha256=result['profile_sha256'],
            window_sample_count=result['sample_count'],elapsed_sec=result['elapsed_sec'])
    return estimate,audit
