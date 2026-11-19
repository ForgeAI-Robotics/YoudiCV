"""Same-frame numeric-layout shadow localization, separated from motion/lamps.

Never append inferred labels to YOLO detections, set a model confidence, call a
robot, or publish an ordinary pose. The existing RGB-D estimator is reused on
this exact frame and its independently detected panel references.
"""
from copy import deepcopy
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np

from .button_pose import ButtonDetection, estimate_button_camera_pose, select_button
from .numeric_layout_geometry import infer_layout, recover_one_conflict
from .numeric_yolo_consistency import inspect_numeric_target
from .observation_visibility import complete_button_box


class NumericLayoutLocalizer:
    def __init__(self, profile, camera_matrix, geometry_options, *, target_labels,
                 profile_sha256, target_confidence=.70, max_frame_age_sec=.35,
                 yolo_consistency_enabled=False):
        self.profile=deepcopy(profile)
        self.profile_sha256=str(profile_sha256)
        self.coordinates=deepcopy(profile['coordinates'])
        self.targets=frozenset(map(str,target_labels))
        self.camera_matrix=np.asarray(camera_matrix,dtype=float).copy()
        self.geometry_options=dict(geometry_options)
        self.target_confidence=float(target_confidence)
        self.max_frame_age_sec=float(max_frame_age_sec)
        self.yolo_consistency_enabled=yolo_consistency_enabled is True
        if profile.get('schema')!=1 or not profile.get('id'):
            raise ValueError('invalid numeric layout profile')
        if not self.targets or not self.targets<=set(self.coordinates):
            raise ValueError('numeric layout target labels absent from profile')
        if any(not t.isdigit() or not 1<=int(t)<=16 for t in self.targets):
            raise ValueError('numeric layout supports numeric floor labels only')
        positions=[]
        for label,point in self.coordinates.items():
            if (not str(label).isdigit() or not 1<=int(label)<=16 or len(point)!=2
                    or not np.isfinite(point).all() or point[0] not in (0,1)):
                raise ValueError('invalid numeric layout coordinates')
            positions.append(tuple(point))
        if len(set(positions))!=len(positions):
            raise ValueError('duplicate numeric layout positions')
        if (self.camera_matrix.shape!=(3,3) or not np.isfinite(self.camera_matrix).all()
                or abs(np.linalg.det(self.camera_matrix))<1e-9):
            raise ValueError('invalid camera matrix')
        if not 0<self.max_frame_age_sec<=.35 or not .70<=self.target_confidence<=1:
            raise ValueError('invalid numeric layout freshness/confidence gate')

    @classmethod
    def from_config(cls, settings, config_path, camera_matrix, geometry_options, target_confidence):
        settings=dict(settings or {})
        mode=settings.get('mode','off')
        if mode=='off':return None
        if mode!='shadow':raise ValueError('numeric layout currently supports off/shadow only')
        path=Path(settings['profile_path']).expanduser()
        if not path.is_absolute():path=Path(config_path).resolve().parent/path
        data=path.read_bytes()
        digest=hashlib.sha256(data).hexdigest()
        if digest!=settings.get('profile_sha256'):
            raise ValueError('numeric layout profile SHA256 mismatch')
        return cls(json.loads(data),camera_matrix,geometry_options,
                   target_labels=settings.get('target_labels',[]),profile_sha256=digest,
                   target_confidence=target_confidence,
                   max_frame_age_sec=settings.get('max_frame_age_sec',.35),
                   yolo_consistency_enabled=settings.get('yolo_consistency_enabled') is True)

    def inspect_yolo(self, frame, target, detections):
        if not self.yolo_consistency_enabled or str(target) not in self.targets:
            return {}
        record=inspect_numeric_target(self.coordinates,detections,str(target),
            [frame.color.shape[1],frame.color.shape[0]],
            min_target_confidence=self.target_confidence)
        return dict(record,frame_sequence=int(frame.sequence),target_label=str(target),
                    profile_sha256=self.profile_sha256,
                    capture_time_unix_sec=float(frame.received_unix))

    def status(self):
        return dict(mode='shadow',profile_id=self.profile['id'],profile_sha256=self.profile_sha256,
                    target_labels=sorted(self.targets,key=int),motion_eligible=False,
                    light_state='not_evaluated')

    def localize_2d(self, frame, target, detections, *, now_monotonic=None):
        """Same existing geometry, without reading depth or granting motion."""
        return self.localize(frame, target, detections, now_monotonic=now_monotonic,
                             depth_required=False)

    def localize(self, frame, target, detections, *, now_monotonic=None,
                 depth_required=True):
        started=time.perf_counter()
        target=str(target)
        out=dict(schema=1,source='panel_layout_inferred',mode='shadow',
                 target_label=target,target_detected=False,yolo_confidence=None,
                 motion_eligible=False,light_state='not_evaluated',
                 frame_sequence=int(frame.sequence),rgbd_bundle_sequence=int(frame.sequence),
                 capture_time_unix_sec=float(frame.received_unix),
                 received_monotonic=float(frame.received_monotonic),
                 color_timestamp=int(frame.color_timestamp),depth_timestamp=int(frame.depth_timestamp),
                 pairing='same_rgbd_bundle',profile_id=self.profile['id'],
                 profile_sha256=self.profile_sha256,accepted_2d=False,valid_3d=False,
                 localization_stage='completed_rgbd' if depth_required else '2d_only')
        try:
            if target not in self.targets:
                out['reason']='target_outside_configured_layout_scope';return out
            try:
                select_button(list(detections),target,self.target_confidence)
                out['reason']='existing_yolo_target_available';return out
            except RuntimeError:pass
            if any(d.label==target and d.confidence>=self.target_confidence for d in detections):
                out['reason']='ambiguous_strong_target_not_overridden';return out
            now=time.monotonic() if now_monotonic is None else float(now_monotonic)
            age=now-float(frame.received_monotonic);out['frame_age_at_processing_sec']=age
            if not -.05<=age<=self.max_frame_age_sec:
                out['reason']='rgbd_frame_stale';return out
            image=frame.color
            if image.ndim!=3 or image.shape[2]!=3:
                out['reason']='rgb_depth_shape_mismatch';return out
            if depth_required:
                depth=np.asarray(frame.depth_m)
                if depth.ndim!=2 or depth.shape!=image.shape[:2]:
                    out['reason']='rgb_depth_shape_mismatch';return out
            out['image_size']=[int(image.shape[1]),int(image.shape[0])]
            anchors=[]
            for d in detections:
                x1,y1,x2,y2=map(float,d.bbox_xyxy)
                if (d.label not in self.coordinates or not math.isfinite(d.confidence)
                        or not self.target_confidence<=d.confidence<=1
                        or not np.isfinite([x1,y1,x2,y2]).all()
                        or min(x1,y1,image.shape[1]-x2,image.shape[0]-y2)<5
                        or min(x2-x1,y2-y1)<14):continue
                anchors.append(dict(label=d.label,center=list(d.center_pixel),
                                    diameter=(x2-x1+y2-y1)/2,confidence=float(d.confidence)))
            # A duplicated neighbour label is unusable as a reference, but must
            # not veto all the other independent, unambiguous neighbours.
            counts=Counter(a['label'] for a in anchors)
            ambiguous={label for label,count in counts.items() if count>1}
            out['reference_filter']=dict(ambiguous_labels=sorted(ambiguous,key=int),
                                         discarded_count=sum(counts[l] for l in ambiguous))
            anchors=[a for a in anchors if a['label'] not in ambiguous]
            result=infer_layout(self.coordinates,anchors,target,image)
            if not result['accepted']:
                recovery=recover_one_conflict(self.coordinates,anchors,target,image)
                if recovery['accepted']:
                    recovery['previous_failure']=result['reason'];result=recovery
            out['geometry']=result
            if not result['accepted']:
                out['reason']=result['reason'];return out
            center=np.asarray(result['center'],float);diameter=float(result['estimated_diameter_px'])
            box=[float(center[0]-diameter/2),float(center[1]-diameter/2),
                 float(center[0]+diameter/2),float(center[1]+diameter/2)]
            out.update(accepted_2d=True,center_xy=center.tolist(),bbox_xyxy=box)
            if not depth_required:
                if not complete_button_box(box, out['image_size']):
                    out.update(accepted_2d=False, reason='inferred_light_region_incomplete')
                    return out
                out['reason']='same_frame_layout_region_available'
                return out
            kept=set(result['anchors'])
            supports=tuple(d for d in detections if d.label in kept and d.confidence>=self.target_confidence)
            # This temporary geometry ROI is never added to sample.detections.
            # Zero explicitly means NO YOLO target confidence, not a weak label.
            roi=ButtonDetection(-1,target,0.0,tuple(box))
            options=dict(self.geometry_options)
            options['min_panel_support_detections']=max(3,int(options.get('min_panel_support_detections',0)))
            pose=estimate_button_camera_pose(roi,depth,self.camera_matrix,
                                            panel_support_detections=supports,**options)
            if not np.isfinite(pose.camera_to_button).all():raise RuntimeError('nonfinite layout pose')
            out.update(valid_3d=True,reason='same_frame_layout_pose_available',
                       pose=dict(p_camera_m=pose.p_camera.tolist(),normal_camera=pose.normal_camera.tolist(),
                                 camera_to_button=pose.camera_to_button.tolist(),plane_rms_m=float(pose.plane_rms_m),
                                 panel_plane_fit=getattr(pose,"panel_plane_fit",None),
                                 plane_inliers=int(pose.plane_inliers),panel_support_detections=int(pose.panel_support_detections),
                                 local_global_depth_delta_m=float(pose.local_global_depth_delta_m)))
        except Exception as exc:
            out.update(valid_3d=False,reason='layout_localization_failed',error=f'{type(exc).__name__}: {exc}')
        finally:
            out['processing_sec']=time.perf_counter()-started
            if 'frame_age_at_processing_sec' in out:
                age_end=out['frame_age_at_processing_sec']+out['processing_sec']
                out['frame_age_at_completion_sec']=age_end
                if out.get('valid_3d') and not -.05<=age_end<=self.max_frame_age_sec:
                    out.update(valid_3d=False,reason='rgbd_frame_stale_after_localization')
                if not depth_required and out.get('accepted_2d') and not -.05<=age_end<=self.max_frame_age_sec:
                    out.update(accepted_2d=False,reason='rgbd_frame_stale_after_2d_localization')
        return out
