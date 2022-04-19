#!/usr/bin/env bash
source activate sgg

LR=0.01
LRI=20
EPOCHS=60
BATCH_SIZE=16

MODEL_TYPE='image_gcn_med_7f_pt'
ATTEN_TYPE='fpam'

NODES_NUM=8
DATASET_NAME='Places365-7'   # Places365-7, Places365-14
NUM_CLASSES=7

LOG_DIR="./logs/""$DATASET_NAME""/"


current_time=$(date  "+%Y%m%d-%H%M%S-")

LOG_FILE1=""$LOG_DIR""$current_time""$MODEL_TYPE""_"$ATTEN_TYPE"_""$DATASET_NAME""_""$BATCH_SIZE""_""$NODES_NUM""_1.txt""

echo $LOG_FILE1


CUDA_VISIBLE_DEVICES=5 python train_gnn_sr_single_gpu.py --model_type $MODEL_TYPE --epochs $EPOCHS --lr $LR --lri $LRI --experiment_id 2 --nodes_num $NODES_NUM --bs $BATCH_SIZE --dataset_name $DATASET_NAME --num_classes $NUM_CLASSES --atten_type $ATTEN_TYPE --fusion True  --status train >> $LOG_FILE1 &
