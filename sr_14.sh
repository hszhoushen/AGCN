#!/usr/bin/env bash
source activate llod

LR=0.01
LRI=20
EPOCHS=60
BATCH_SIZE=16
NODES_NUM=20
NUM_CLASSES=14

MODEL_TYPE='image_gcn_med_14'
DATASET_NAME='Places365-14'   # esc10, esc50, US8K
LOG_DIR="./logs/""$DATASET_NAME""/"
current_time=$(date  "+%Y%m%d-%H%M%S-")
ATTEN_TYPE='pafm'

LOG_FILE1=""$LOG_DIR""$current_time""$MODEL_TYPE""_"$ATTEN_TYPE"_""$DATASET_NAME""_""$BATCH_SIZE""_""$NODES_NUM""_1.txt""

echo $LOG_FILE1


CUDA_VISIBLE_DEVICES=1 python train_gnn_sr.py --model_type $MODEL_TYPE --epochs $EPOCHS --lr $LR --lri $LRI --experiment_id 14 --nodes_num $NODES_NUM --bs $BATCH_SIZE --dataset_name $DATASET_NAME --atten_type $ATTEN_TYPE --num_classes $NUM_CLASSES >> $LOG_FILE1 &
