#!/bin/bash

cd /data/jiali/vlul
python -m torch.distributed.run --nproc_per_node=2 train.py --cfg-path configs/original/albef/retrieval_coco.yaml
# CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.run --nproc_per_node=2 train.py --cfg-path configs/original/blip/retrieval_coco.yaml

python -m torch.distributed.run --nproc_per_node=2 train.py --cfg-path configs/original/albef/vqa.yaml
# CUDA_VISIBLE_DEVICES=0,2 python -m torch.distributed.run --nproc_per_node=2 train.py --cfg-path configs/original/blip/vqa.yaml


python -m torch.distributed.run --nproc_per_node=2 train.py --cfg-path configs/original/blip/nlvr.yaml
# CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.run --nproc_per_node=2 train.py --cfg-path configs/original/albef/nlvr.yaml

python -m torch.distributed.run --nproc_per_node=2 train.py --cfg-path configs/original/albef/snli_ve.yaml
# CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.run --nproc_per_node=2 train.py --cfg-path configs/original/blip/snli_ve.yaml
