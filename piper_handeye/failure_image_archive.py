"""Bounded in-memory original images, persisted only for failed tasks/attempts."""
from __future__ import annotations

from collections import deque
import copy
import hashlib
import json
from pathlib import Path
import re
import threading

MAX_IMAGE_BYTES = 2 * 1024 * 1024
MAX_BUCKETS = 16
MAX_PER_BUCKET = 16
MAX_BUFFER_BYTES = 12 * 1024 * 1024


def _json_default(value):
    if hasattr(value, 'tolist'): return value.tolist()
    if hasattr(value, 'item'): return value.item()
    raise TypeError(type(value).__name__)


class FailureImageArchive:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.attempt = 0
        self.failed_attempts = set()
        self.buckets = {}
        self.errors = deque(maxlen=12)
        self.lock = threading.Lock()
        self.seen = 0
        self.buffer_bytes = 0

    def begin_attempt(self, attempt):
        attempt = int(attempt)
        with self.lock:
            if self.attempt and attempt > self.attempt:
                self.failed_attempts.add(self.attempt)
            self.attempt = attempt

    def capture(self, entry):
        """Copy JPEG bytes before the 20-frame temporary cache can prune them.

        No inference, image decode, crop, or disk writes in the observation path.
        Failure capture errors are diagnostics, never an ON/OFF or motion gate.
        """
        try:
            path = Path(entry['image'])
            if path.stat().st_size > MAX_IMAGE_BYTES:
                raise ValueError('image exceeds bounded archive size')
            raw = path.read_bytes()
            # Keep original bytes, metadata and sequence together. Crop later.
            metadata = copy.deepcopy(entry)
            position = str(entry.get('position', 'unknown'))
            group = ('initial' if position == 'waiting' else
                     'before_press' if position == 'before_press' else
                     'post_press' if position.startswith(('near','prepress','press_monitor')) else
                     'waiting_off')
            with self.lock:
                key = (self.attempt, group)
                if key not in self.buckets:
                    if len(self.buckets) >= MAX_BUCKETS:
                        raise ValueError('archive bucket limit exceeded')
                    self.buckets[key] = deque(maxlen=MAX_PER_BUCKET)
                bucket = self.buckets[key]
                if len(bucket)==bucket.maxlen:
                    # Keep the first two baseline/transition frames as well as
                    # the latest fourteen, rather than losing the beginning.
                    self.buffer_bytes -= len(bucket[2][1])
                    del bucket[2]
                bucket.append((metadata, raw))
                self.buffer_bytes += len(raw)
                while self.buffer_bytes > MAX_BUFFER_BYTES:
                    oldest_key = next(k for k,items in self.buckets.items() if items)
                    self.buffer_bytes -= len(self.buckets[oldest_key].popleft()[1])
                self.seen += 1
        except Exception as exc:
            with self.lock:
                self.errors.append(dict(sequence=entry.get('frame_sequence'),
                    reason=str(exc), image=entry.get('image')))

    def finalize(self, *, success, context):
        with self.lock:
            failed = set(self.failed_attempts)
            if not success:
                failed.add(self.attempt)
            # Initial view belongs to any subsequent failed attempt/task too.
            selected = [(key, list(rows)) for key,rows in self.buckets.items()
                        if failed and (key[0] in failed or key[0]==0)]
            errors = list(self.errors)
            seen = self.seen
        if not failed:
            return dict(policy='failure_only', retained=False, buffered_frames=seen,
                        reason='verified_success', capture_errors=errors)
        import cv2
        import numpy as np
        self.directory.mkdir(parents=True, exist_ok=True)
        records = []
        for (attempt, group), rows in selected:
            for metadata, raw in rows:
                seq = metadata.get('frame_sequence')
                stem = 'attempt_%d_%s_frame_%s' % (attempt,group,re.sub(r'[^0-9A-Za-z_-]','_',str(seq)))
                original = self.directory/(stem+'_full.jpg')
                temporary=original.with_suffix('.tmp');temporary.write_bytes(raw);temporary.replace(original)
                record=dict(observation=metadata, original_image=str(original),
                            sha256=hashlib.sha256(raw).hexdigest(), crop_image=None)
                region = metadata.get('target_region') or metadata.get('target_detection') or {}
                box = region.get('bbox_xyxy')
                try:
                    if not box or region.get('frame_sequence')!=seq:
                        raise ValueError('no same-frame region; original retained without an invented crop')
                    image=cv2.imdecode(np.frombuffer(raw,dtype=np.uint8),cv2.IMREAD_COLOR)
                    if image is None:raise ValueError('original image decode failed')
                    x1,y1,x2,y2=np.rint(np.asarray(box,dtype=float)).astype(int)
                    h,w=image.shape[:2]
                    if not (0<=x1<x2<=w and 0<=y1<y2<=h):raise ValueError('crop outside original')
                    crop=self.directory/(stem+'_button.png')
                    if not cv2.imwrite(str(crop),image[y1:y2,x1:x2]):raise RuntimeError('crop write failed')
                    record.update(crop_image=str(crop),crop_xyxy=[int(v) for v in (x1,y1,x2,y2)],
                                  crop_policy='same_frame_original_pixels_no_enhancement')
                except Exception as exc:
                    record['crop_error']=str(exc)
                records.append(record)
        manifest=dict(policy='failure_only',context=context,failed_attempts=sorted(failed),
                      buffered_frames=seen,saved_frames=len(records),capture_errors=errors,records=records)
        path=self.directory/'failure_record.json';temporary=path.with_suffix('.tmp')
        temporary.write_text(json.dumps(manifest,ensure_ascii=False,indent=2,default=_json_default)+'\n')
        temporary.replace(path)
        return dict(policy='failure_only',retained=True,directory=str(self.directory),
                    record=str(path),saved_frames=len(records),failed_attempts=sorted(failed),capture_errors=errors)


def remove_success_images(run_directory):
    """Only newly completed run-owned images; never historical runs or symlinks."""
    root=Path(run_directory).resolve();removed=0
    for path in root.rglob('*'):
        if (path.is_file() and not path.is_symlink() and path.suffix.lower() in {'.jpg','.jpeg','.png'}
                and root in path.resolve().parents):
            path.unlink();removed+=1
    return removed
