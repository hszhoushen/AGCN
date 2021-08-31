#!/usr/bin/env bash
source activate llod


BATCH_SIZE=16

MODEL_TYPE='image_gcn_med_7f_pt'
ATTEN_TYPE='pafm'

MODEL_NAME='image_gcn_med_7f_pt_pafm_2_16_epoch_20.pth.tar'
NODES_NUM=16
DATASET_NAME='Places365-7'   # Places365-7, Places365-14, SUNRGBD
NUM_CLASSES=7



CUDA_VISIBLE_DEVICES=0 python train_gnn_sr.py --model_type $MODEL_TYPE --experiment_id 2 --nodes_num $NODES_NUM --model_name $MODEL_NAME --bs $BATCH_SIZE --dataset_name $DATASET_NAME --num_classes $NUM_CLASSES --atten_type $ATTEN_TYPE --fusion False --status draw
