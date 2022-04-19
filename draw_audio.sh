#!/usr/bin/env bash
source activate llod


BATCH_SIZE=16

MODEL_TYPE='image_gcn_med_7f_pt'
ATTEN_TYPE='fpam'


# 95.0, 'audio_gcn_max_med_fpam_7f_20_epoch_44_20.pth.tar'
# 93.75, 'audio_gcn_max_med_fpam_7f_20_epoch_22_20.pth.tar'
# 83.75, 'audio_gcn_max_med_fpam_7f_20_epoch_6_20.pth.tar'
# 75.0, 'audio_gcn_max_med_fpam_7f_20_epoch_4_20.pth.tar'
# 51.25, 'audio_gcn_max_med_fpam_7f_20_epoch_2_20.pth.tar'

MODEL_NAME='audio_gcn_max_med_fpam_7f_20_epoch_4_20.pth.tar'

NODES_NUM=20
DATASET_NAME='ESC10'   # ESC10, ESC50, US8K
NUM_CLASSES=50
CUDA_VISIBLE_DEVICES=4 python train_gnn_sr.py --model_type $MODEL_TYPE --experiment_id 2 --nodes_num $NODES_NUM --model_name $MODEL_NAME --bs $BATCH_SIZE --dataset_name $DATASET_NAME --num_classes $NUM_CLASSES --atten_type $ATTEN_TYPE --fusion True --status draw_audio --pretrain
