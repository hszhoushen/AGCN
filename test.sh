#!/usr/bin/env bash
source activate llod


BATCH_SIZE=16


MODEL_TYPE='image_gcn_med_7f_pt'
ATTEN_TYPE='pafm'

MODEL_NAME='image_gcn_med_7f_pt_pafm_2_20_epoch_4.pth.tar' #'image_gcn_med_7f_pt_pafm_2_16_epoch_6.pth.tar' #'image_gcn_med_7f_pt_pafm_2_16_epoch_15.pth.tar'
NODES_NUM=20
DATASET_NAME='Places365-7'   # Places365-7, Places365-14, SUNRGBD
NUM_CLASSES=7

LOG_DIR="./logs/""$DATASET_NAME""/"



CUDA_VISIBLE_DEVICES=9 python train_gnn_sr.py --model_type $MODEL_TYPE --model_name $MODEL_NAME --experiment_id 2 --nodes_num $NODES_NUM --bs $BATCH_SIZE --dataset_name $DATASET_NAME --num_classes $NUM_CLASSES --atten_type $ATTEN_TYPE --fusion True --status test
