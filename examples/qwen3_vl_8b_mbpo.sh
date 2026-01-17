#!/bin/bash

set -x

MODEL_PATH=Qwen/Qwen3-VL-8B-Instruct
ROLLOUT_BS=512
GEN_BS=128
GLOBAL_BS=128
MB_PER_UPDATE=4
MB_PER_EXP=8

python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=hiyouga/geometry3k@train \
    data.val_files=hiyouga/geometry3k@test \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.actor.clip_ratio_low=0.2 \
    worker.actor.clip_ratio_high=0.28 \
    worker.rollout.n=4 \
    worker.actor.optim.lr=1.0e-6 \
    algorithm.disable_kl=False \
    algorithm.use_kl_loss=True \
    algorithm.online_filtering=True \
    algorithm.adv_estimator=mbpo \
    trainer.experiment_name=qwen3_vl_8b_geo_mbpo \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=8 \
    data.rollout_batch_size="${ROLLOUT_BS}" \
    data.mini_rollout_batch_size="${GEN_BS}" \
    worker.actor.global_batch_size="${GLOBAL_BS}" \
    worker.actor.micro_batch_size_per_device_for_update="${MB_PER_UPDATE}" \
    worker.actor.micro_batch_size_per_device_for_experience="${MB_PER_EXP}" \
    worker.rollout.tensor_parallel_size=2
