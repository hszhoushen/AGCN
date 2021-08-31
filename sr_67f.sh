#!/usr/bin/env bash
source activate llod

LR=0.01
LRI=10
EPOCHS=60
BATCH_SIZE=16
NODES_NUM=8
NUM_CLASSES=67

MODEL_TYPE='image_gcn_med_67f'
DATASET_NAME='MIT67'   # esc10, esc50, US8K, Places365-7, Places365-14, MIT67
ATTEN_TYPE='pafm'

LOG_DIR="./logs/""$DATASET_NAME""/"
EID=67

current_time=$(date  "+%Y%m%d-%H%M%S-")

LOG_FILE1=""$LOG_DIR""$current_time""$MODEL_TYPE""_"$ATTEN_TYPE"_""$DATASET_NAME""_""$BATCH_SIZE""_""$NODES_NUM""_1.txt""

echo $LOG_FILE1


CUDA_VISIBLE_DEVICES=4 python train_gnn_sr.py --model_type $MODEL_TYPE --epochs $EPOCHS --lr $LR --lri $LRI --experiment_id $EID --nodes_num $NODES_NUM --bs $BATCH_SIZE --dataset_name $DATASET_NAME --num_classes $NUM_CLASSES --atten_type $ATTEN_TYPE --fusion True >> $LOG_FILE1 &
