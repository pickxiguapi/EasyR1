#!/bin/bash
set -x

DATE=$(date +%Y%m%d)
NODE_NAME=$(hostname)
EXP_NAME=20260211_Embodied-R1.5_MBPO_only_image
LOG_DIR=logs
mkdir -p ${LOG_DIR}
LOG_FILE=${LOG_DIR}/${DATE}_${NODE_NAME}_${EXP_NAME}.log

TRAIN_FILES="[rft_train_datasets/ER1.5_CoSyn-point_image_point.json,rft_train_datasets/ER1.5_Droid-Trace_image_trace.json,rft_train_datasets/ER1.5_EO_image_qa.json,rft_train_datasets/ER1.5_ER1-point_image_point.json,rft_train_datasets/ER1.5_ER1-trace_image_trace.json,rft_train_datasets/ER1.5_ERQA2_image_qa.json,rft_train_datasets/ER1.5_ERQA_Rush_image_qa.json,rft_train_datasets/ER1.5_general_image_qa_filtered.json,rft_train_datasets/ER1.5_HandAL_image_point.json,rft_train_datasets/ER1.5_HOI4D-Trace_image_trace.json,rft_train_datasets/ER1.5_InstructPart_image_point.json,rft_train_datasets/ER1.5_InternData-Trace_image_trace.json,rft_train_datasets/ER1.5_Ref_L4_image_point.json,rft_train_datasets/ER1.5_Refspatial_image_point.json,rft_train_datasets/ER1.5_regular_simulation_image_point.json,rft_train_datasets/ER1.5_regular_synthetic_image_point.json,rft_train_datasets/ER1.5_Robo2VLM_image_qa.json,rft_train_datasets/ER1.5_robocasa_partnet_2d_image_trace.json,rft_train_datasets/ER1.5_robocasa_partnet_3d_image_trace.json,rft_train_datasets/ER1.5_Roborefit_image_point.json,rft_train_datasets/ER1.5_RoboVQA_image.json,rft_train_datasets/ER1.5_SAT_image_qa.json,rft_train_datasets/ER1.5_spatialssrl_image_qa.json,rft_train_datasets/ER1.5_Temporal_image_qa.json]"
TEST_FILES="[rft_test_datasets/erqa.json,rft_test_datasets/refspatial.json,rft_test_datasets/sat.json,rft_test_datasets/vabench_p.json,rft_test_datasets/where2place.json]"

CONFIG=ER1.5_scripts/20260211_Embodied-R1.5_MBPO.yaml
MODEL_PATH=/apdcephfs_wza/cientgu/iffyuan/models/Embodied-R1.5-SFT-0128
IMAGE_DIR=/apdcephfs_wza/cientgu/iffyuan/Embodied-R1.5-RFT/data
ROLLOUT_BS=1024
GEN_BS=256
GLOBAL_BS=512
MB_PER_UPDATE=4
MB_PER_EXP=8
REWARD=ER1.5_scripts/reward_function/embodied_reward.py:compute_score

python3 -m verl.trainer.main \
    config=${CONFIG} \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.actor.clip_ratio_low=0.2 \
    worker.actor.clip_ratio_high=0.28 \
    worker.rollout.n=8 \
    algorithm.adv_estimator=mbpo \
    trainer.experiment_name=${EXP_NAME} \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=2 \
    trainer.val_freq=20 \
    trainer.save_freq=15 \
    trainer.save_limit=2 \
    data.rollout_batch_size="${ROLLOUT_BS}" \
    data.mini_rollout_batch_size="${GEN_BS}" \
    data.train_files=$TRAIN_FILES \
    data.val_files=$TEST_FILES \
    data.image_dir=$IMAGE_DIR \
    worker.actor.global_batch_size="${GLOBAL_BS}" \
    worker.actor.micro_batch_size_per_device_for_update="${MB_PER_UPDATE}" \
    worker.actor.micro_batch_size_per_device_for_experience="${MB_PER_EXP}" \
    worker.rollout.tensor_parallel_size=4 \
    worker.reward.reward_function=${REWARD} \
    2>&1 | tee ${LOG_FILE}
