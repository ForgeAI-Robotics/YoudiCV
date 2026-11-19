"""Fixed-length, fixed-strata 2x hard-invalid sampling; local trainer only.

No installed-package edits. Val/test loader behavior remains the original one.
Sampler draw receipts can be prefetched; epoch_exposure records actual batches.
"""
import hashlib
import json
import os
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import Sampler
from ultralytics.data.build import InfiniteDataLoader,seed_worker
from ultralytics.models.yolo.pose import PoseTrainer
from ultralytics.utils import RANK
from ultralytics.utils.torch_utils import torch_distributed_zero_first

ROOT=Path(__file__).resolve().parent
DATA=ROOT/'expansion_v7/yolo26_mix25_25_50'
WORK=ROOT/'expansion_v7/hard_invalid_v7c'
TARGET=dict(normal=500,occlusion=250,geometry=250,negative=280)


class FixedStrataHardSampler(Sampler):
    def __init__(self,paths,rows,selected,seed=20261015,receipt_dir=None):
        self.paths=[str(Path(p).resolve()) for p in paths]
        self.rows={str(Path(r['image']).resolve()):r for r in rows}
        self.hard={str(Path(r['image']).resolve()) for r in selected}
        self.groups={g:[] for g in TARGET};self.seed=seed;self.draw_number=0
        self.receipt_dir=Path(receipt_dir) if receipt_dir else None
        if self.receipt_dir:self.receipt_dir.mkdir(parents=True,exist_ok=True)
        assert len(self.paths)==1280 and len(set(self.paths))==1280
        for i,p in enumerate(self.paths):
            r=self.rows[p];assert r['split']=='train'
            self.groups[r['group']].append(i)
        assert {g:len(ids) for g,ids in self.groups.items()}==TARGET
        assert self.hard<=set(self.paths)
        for g,ids in self.groups.items():
            h=sum(self.paths[i] in self.hard for i in ids)
            assert h<=len(ids)//10
            if g in ['negative','occlusion']:assert h==0

    def indices_for_epoch(self,epoch):
        generator=torch.Generator().manual_seed(self.seed+epoch)
        indices=[]
        for group,ids in self.groups.items():
            hard=[i for i in ids if self.paths[i] in self.hard]
            ordinary=[i for i in ids if self.paths[i] not in self.hard]
            count=len(ids)-2*len(hard)
            keep=torch.randperm(len(ordinary),generator=generator).tolist()[:count]
            indices.extend([ordinary[i] for i in keep]+hard+hard)
        order=torch.randperm(len(indices),generator=generator).tolist()
        return [indices[i] for i in order]

    def __iter__(self):
        draw=self.draw_number;self.draw_number+=1
        indices=self.indices_for_epoch(draw)
        counts=Counter(indices)
        assert Counter(self.rows[self.paths[i]]['group'] for i in indices)==TARGET
        assert all(counts[i]==2 for i,p in enumerate(self.paths) if p in self.hard)
        if self.receipt_dir:
            path=self.receipt_dir/f'draw_{draw+1:03d}.json'
            assert not path.exists(),'Refuse to overwrite sampler receipt'
            path.write_text(json.dumps(dict(draw_number=draw+1,seed=self.seed+draw,
                note='May be prefetched; use epoch_exposure for actually consumed epochs',
                group_counts=TARGET,unique_images=len(counts),hard_images=len(self.hard),
                indices=indices),ensure_ascii=False))
        return iter(indices)

    def __len__(self):return 1280


class HardInvalidPoseTrainer(PoseTrainer):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.exposure_epoch=None;self.exposure_paths=[]
        self.callbacks['on_train_epoch_end'].append(self.audit_exposure)

    def get_dataloader(self,dataset_path,batch_size=16,rank=0,mode='train'):
        if mode!='train':return super().get_dataloader(dataset_path,batch_size,rank,mode)
        assert rank==-1 and self.world_size<=1,'This experimental sampler is single-GPU only'
        with torch_distributed_zero_first(rank):dataset=self.build_dataset(dataset_path,mode,batch_size)
        audit=json.loads((DATA/'audit.json').read_text())
        selection=json.loads((WORK/'hard_selection.json').read_text())
        assert hashlib.sha256((DATA/'audit.json').read_bytes()).hexdigest()==selection['data_audit_sha256']
        self.train_rows={r['image']:r for r in audit['files'] if r['split']=='train'}
        self.hard_set={r['image'] for r in selection['records']}
        self.hard_sampler=FixedStrataHardSampler(dataset.im_files,list(self.train_rows.values()),
            selection['records'],seed=self.args.seed,receipt_dir=WORK/'sampler_draws')
        generator=torch.Generator().manual_seed(6148914691236517205+RANK)
        workers=min(os.cpu_count(),self.args.workers)
        loader=InfiniteDataLoader(dataset=dataset,batch_size=min(batch_size,len(dataset)),
            shuffle=False,num_workers=workers,sampler=self.hard_sampler,prefetch_factor=4 if workers else None,
            pin_memory=self.device.type=='cuda',collate_fn=dataset.collate_fn,
            worker_init_fn=seed_worker,generator=generator,drop_last=False)
        assert len(loader)==160
        return loader

    def preprocess_batch(self,batch):
        if self.exposure_epoch!=self.epoch:
            self.exposure_epoch=self.epoch;self.exposure_paths=[]
        self.exposure_paths.extend(str(Path(p).resolve()) for p in batch['im_file'])
        return super().preprocess_batch(batch)

    def audit_exposure(self,trainer):
        assert trainer is self
        counts=Counter(self.exposure_paths)
        groups=Counter(self.train_rows[p]['group'] for p in self.exposure_paths)
        assert len(self.exposure_paths)==1280 and groups==TARGET
        assert all(counts[p]==2 for p in self.hard_set)
        assert all(n==1 for p,n in counts.items() if p not in self.hard_set)
        class_counts=Counter()
        for p in self.exposure_paths:
            for line in Path(self.train_rows[p]['label']).read_text().splitlines():
                class_counts[line.split()[0]]+=1
        directory=WORK/'epoch_exposure';directory.mkdir(exist_ok=True)
        path=directory/f'epoch_{self.epoch+1:03d}.json';assert not path.exists()
        path.write_text(json.dumps(dict(epoch=self.epoch+1,consumed_images=1280,
            unique_images=len(counts),hard_images=len(self.hard_set),hard_exposures=sum(counts[p] for p in self.hard_set),
            group_counts=dict(groups),class_instance_exposures=dict(class_counts),
            hard_image_counts={p:counts[p] for p in sorted(self.hard_set)},
            train_sources_only=True),ensure_ascii=False,indent=2))
