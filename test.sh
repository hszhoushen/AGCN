#!/usr/bin/env bash
source activate llod

LR=0.01
LRI=20
EPOCHS=60
BATCH_SIZE=16

MODEL_TYPE='image_gcn_med_7f_pt'
ATTEN_TYPE='fpam'

MODEL_NAME='image_gcn_med_7_pt_fpam_3_12_epoch_9.pth.tar'
#MODEL_NAME='image_gcn_med_7f_pt_fpam_2_24_epoch_9_best.pth.tar' #'image_gcn_med_7f_pt_pafm_2_16_epoch_6.pth.tar' #'image_gcn_med_7f_pt_pafm_2_16_epoch_15.pth.tar'

NODES_NUM=12
DATASET_NAME='Places365-7'   # Places365-7, Places365-14, SUNRGBD
NUM_CLASSES=7

LOG_DIR="./logs/""$DATASET_NAME""/"



CUDA_VISIBLE_DEVICES=0 python train_gnn_sr.py --model_type $MODEL_TYPE --epochs $EPOCHS --lr $LR --lri $LRI --model_name $MODEL_NAME --nodes_num $NODES_NUM --bs $BATCH_SIZE --dataset_name $DATASET_NAME --num_classes $NUM_CLASSES --atten_type $ATTEN_TYPE --fusion True --status test
