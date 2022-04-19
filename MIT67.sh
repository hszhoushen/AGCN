#!/usr/bin/env bash
source activate llod

LR=0.01
LRI=30
EPOCHS=300
BATCH_SIZE=16
NODES_NUM=20
NUM_CLASSES=67

MODEL_TYPE='image_gcn_max_med'
EID='67f'
DATASET_NAME='MIT67'   # esc10, esc50, US8K, Places365-7, Places365-14, MIT67
LOG_DIR="./logs/""$DATASET_NAME""/"
current_time=$(date  "+%Y%m%d-%H%M%S-")

LOG_FILE1=""$LOG_DIR""$current_time""$MODEL_TYPE""_""$EID""_""$DATASET_NAME""_""$BATCH_SIZE""_""$NODES_NUM""_""$LRI"".txt""

echo $LOG_FILE1


CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --master_port 21038 --nproc_per_node=2 train_gnn_sr.py --model_type $MODEL_TYPE --epochs $EPOCHS --lr $LR --lri $LRI --experiment_id $EID --nodes_num $NODES_NUM --batch_size $BATCH_SIZE --dataset_name $DATASET_NAME --num_classes $NUM_CLASSES --fusion True --status train # >> $LOG_FILE1 &
