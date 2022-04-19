#!/usr/bin/env bash
source activate llod

LR=0.01
LRI=20
EPOCHS=100
BATCH_SIZE=32
NODES_NUM=20
NUM_CLASSES=14
EID='14f_pt'

MODEL_TYPE='image_gcn_max_med'
DATASET_NAME='Places365-14'   # esc10, esc50, US8K
LOG_DIR="./logs/""$DATASET_NAME""/"
current_time=$(date  "+%Y%m%d-%H%M%S-")
ATTEN_TYPE='fpam'  # fpam

LOG_FILE1=""$LOG_DIR""$current_time""$MODEL_TYPE""_""$ATTEN_TYPE""_""$DATASET_NAME""_""$BATCH_SIZE""_""$NODES_NUM""_""$LRI"".txt""

echo $LOG_FILE1


CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.launch --master_port 30045  --nproc_per_node=2 train_gnn_sr.py --model_type $MODEL_TYPE --epochs $EPOCHS --lr $LR --lri $LRI --experiment_id $EID --nodes_num $NODES_NUM --batch_size $BATCH_SIZE --dataset_name $DATASET_NAME --atten_type $ATTEN_TYPE --num_classes $NUM_CLASSES --fusion True --status train --pretrain >> $LOG_FILE1 &
