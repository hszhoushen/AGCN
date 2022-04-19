#!/usr/bin/env bash
source activate swin

LR=0.01
LRI=20
EPOCHS=80
NODES_NUM=24
BATCH_SIZE=16
MODEL_TYPE='audio_gcn_max_med'
ATTEN_TYPE='fpam'
ARCH='resnet50-audio'
DATASET_NAME='ESC10'   # ESC10, ESC50, US8K
LOG_DIR="./logs/""$DATASET_NAME""/"
current_time=$(date  "+%Y%m%d-%H%M%S-")
NUM_CLASSES=10

LOG_FILE1=""$LOG_DIR""$current_time""$MODEL_TYPE""_""$DATASET_NAME""_""$BATCH_SIZE""_""$NODES_NUM""_1.txt""
LOG_FILE2=""$LOG_DIR""$current_time""$MODEL_TYPE""_""$DATASET_NAME""_""$BATCH_SIZE""_""$NODES_NUM""_2.txt""
LOG_FILE3=""$LOG_DIR""$current_time""$MODEL_TYPE""_""$DATASET_NAME""_""$BATCH_SIZE""_""$NODES_NUM""_3.txt""
LOG_FILE4=""$LOG_DIR""$current_time""$MODEL_TYPE""_""$DATASET_NAME""_""$BATCH_SIZE""_""$NODES_NUM""_4.txt""
LOG_FILE5=""$LOG_DIR""$current_time""$MODEL_TYPE""_""$DATASET_NAME""_""$BATCH_SIZE""_""$NODES_NUM""_5.txt""

echo $LOG_FILE1,$LOG_FILE2,$LOG_FILE3,$LOG_FILE4,$LOG_FILE5


CUDA_VISIBLE_DEVICES=0 python train_gnn_sr.py --model_type $MODEL_TYPE --epochs $EPOCHS --lr $LR --lri $LRI --test_set_id 1 --nodes_num $NODES_NUM --arch $ARCH --num_classes $NUM_CLASSES --atten_type $ATTEN_TYPE --fusion True --batch_size $BATCH_SIZE  --dataset_name $DATASET_NAME --pretrain >> $LOG_FILE1 &
CUDA_VISIBLE_DEVICES=1 python train_gnn_sr.py --model_type $MODEL_TYPE --epochs $EPOCHS --lr $LR --lri $LRI --test_set_id 2 --nodes_num $NODES_NUM --arch $ARCH --num_classes $NUM_CLASSES --atten_type $ATTEN_TYPE --fusion True --batch_size $BATCH_SIZE  --dataset_name $DATASET_NAME --pretrain >> $LOG_FILE2 &
CUDA_VISIBLE_DEVICES=2 python train_gnn_sr.py --model_type $MODEL_TYPE --epochs $EPOCHS --lr $LR --lri $LRI --test_set_id 3 --nodes_num $NODES_NUM --arch $ARCH --num_classes $NUM_CLASSES --atten_type $ATTEN_TYPE --fusion True --batch_size $BATCH_SIZE  --dataset_name $DATASET_NAME --pretrain >> $LOG_FILE3 &
CUDA_VISIBLE_DEVICES=3 python train_gnn_sr.py --model_type $MODEL_TYPE --epochs $EPOCHS --lr $LR --lri $LRI --test_set_id 4 --nodes_num $NODES_NUM --arch $ARCH --num_classes $NUM_CLASSES --atten_type $ATTEN_TYPE --fusion True --batch_size $BATCH_SIZE  --dataset_name $DATASET_NAME --pretrain >> $LOG_FILE4 &
CUDA_VISIBLE_DEVICES=4 python train_gnn_sr.py --model_type $MODEL_TYPE --epochs $EPOCHS --lr $LR --lri $LRI --test_set_id 5 --nodes_num $NODES_NUM --arch $ARCH --num_classes $NUM_CLASSES --atten_type $ATTEN_TYPE --fusion True --batch_size $BATCH_SIZE  --dataset_name $DATASET_NAME --pretrain >> $LOG_FILE5 &
