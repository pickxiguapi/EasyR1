#!/bin/bash

set -x

MODEL_PATH=Qwen/Qwen3-VL-8B-Instruct

python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=hiyouga/geometry3k@train \
    data.val_files=hiyouga/geometry3k@test \
    data.mini_rollout_batch_size=128 \
    data.prompt_key=problem \
    data.answer_key=answer \
    data.image_key=images \
    data.video_key=videos \
    data.max_prompt_length=2048 \
    data.max_response_length=2048 \
    data.rollout_batch_size=512 \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.actor.clip_ratio_low=0.2 \
    worker.actor.clip_ratio_high=0.28 \
    worker.rollout.n=5 \
    worker.actor.optim.lr=1.0e-6 \
    algorithm.disable_kl=True \
    algorithm.online_filtering=True \
    algorithm.adv_estimator=mbpo \
    trainer.experiment_name=qwen3_vl_8b_geo_mbpo \
    trainer.n_gpus_per_node=8
