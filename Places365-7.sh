#!/usr/bin/env bash
source activate llod

LR=0.01
LRI=20
EPOCHS=100

BATCH_SIZE=32
NODES_NUM=20
NUM_CLASSES=7
MODEL_TYPE='image_gcn_max_med'
arch='resnet18'
EID='7f_pt'
DATASET_NAME='Places365-7'   # Places365-7, Places365-14
LOG_DIR="./logs/""$DATASET_NAME""/"
current_time=$(date  "+%Y%m%d-%H%M%S-")
ATTEN_TYPE='fpam'    # fpam, afm

LOG_FILE1=""$LOG_DIR""$current_time""$MODEL_TYPE""_"$ATTEN_TYPE"_""$DATASET_NAME""_""$BATCH_SIZE""_""$NODES_NUM""_1.txt""

echo $LOG_FILE1


CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --master_port 20045 --nproc_per_node=2 train_gnn_sr.py --model_type $MODEL_TYPE --epochs $EPOCHS --arch $arch --lr $LR --lri $LRI --experiment_id $EID --nodes_num $NODES_NUM --batch_size $BATCH_SIZE --dataset_name $DATASET_NAME --num_classes $NUM_CLASSES --atten_type $ATTEN_TYPE --pretrain --fusion True --status train >> $LOG_FILE1 &

