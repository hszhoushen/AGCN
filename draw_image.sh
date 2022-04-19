#!/usr/bin/env bash
source activate llod


BATCH_SIZE=16

MODEL_TYPE='image_gcn_med_7f_pt'
ATTEN_TYPE='fpam'


# 'image_gcn_max_med_fpam_7f_pf_20_epoch_40_20.pth.tar', 91.28%
# 'image_gcn_max_med_fpam_7f_pf_20_epoch_24_20.pth.tar', 90.57%
# 'image_gcn_max_med_fpam_7f_pf_20_epoch_12_20.pth.tar', 87.85%
# 'image_gcn_max_med_fpam_7f_pf_20_epoch_4_20.pth.tar', 86.71%
# 'image_gcn_max_med_fpam_7f_pf_20_epoch_2_20.pth.tar', 86.28%
MODEL_NAME='image_gcn_max_med_afm_14f_pf_20_epoch_24_20.pth.tar'

NODES_NUM=20
DATASET_NAME='Places365-14'   # Places365-7, Places365-14, SUNRGBD
NUM_CLASSES=14



CUDA_VISIBLE_DEVICES=9 python train_gnn_sr.py --model_type $MODEL_TYPE --experiment_id 2 --nodes_num $NODES_NUM --model_name $MODEL_NAME --bs $BATCH_SIZE --dataset_name $DATASET_NAME --num_classes $NUM_CLASSES --atten_type $ATTEN_TYPE --fusion True --status draw_image --pretrain
