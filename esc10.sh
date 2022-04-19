#!/usr/bin/env bash
source activate llod

LR=0.01
LRI=20
EPOCHS=300
NODES_NUM=24
BATCH_SIZE=16
MODEL_TYPE='audio_gcn_max_med'
ATTEN_TYPE='fpam'
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


#CUDA_VISIBLE_DEVICES=0 python train_gnn_sr.py --model_type $MODEL_TYPE --epochs $EPOCHS --lr $LR --lri $LRI --test_set_id 1 --nodes_num $NODES_NUM --num_classes $NUM_CLASSES --atten_type $ATTEN_TYPE --fusion True --batch_size $BATCH_SIZE  --dataset_name $DATASET_NAME --pretrain >> $LOG_FILE1 &
#CUDA_VISIBLE_DEVICES=1 python train_gnn_sr.py --model_type $MODEL_TYPE --epochs $EPOCHS --lr $LR --lri $LRI --test_set_id 2 --nodes_num $NODES_NUM --num_classes $NUM_CLASSES --atten_type $ATTEN_TYPE --fusion True --batch_size $BATCH_SIZE  --dataset_name $DATASET_NAME --pretrain >> $LOG_FILE2 &
#CUDA_VISIBLE_DEVICES=3 python train_gnn_sr.py --model_type $MODEL_TYPE --epochs $EPOCHS --lr $LR --lri $LRI --test_set_id 3 --nodes_num $NODES_NUM --num_classes $NUM_CLASSES --atten_type $ATTEN_TYPE --fusion True --batch_size $BATCH_SIZE  --dataset_name $DATASET_NAME --pretrain >> $LOG_FILE3 &
CUDA_VISIBLE_DEVICES=0 python train_gnn_sr.py --model_type $MODEL_TYPE --epochs $EPOCHS --lr $LR --lri $LRI --test_set_id 4 --nodes_num $NODES_NUM --num_classes $NUM_CLASSES --atten_type $ATTEN_TYPE --fusion True --batch_size $BATCH_SIZE  --dataset_name $DATASET_NAME --pretrain >> $LOG_FILE4 &
CUDA_VISIBLE_DEVICES=3 python train_gnn_sr.py --model_type $MODEL_TYPE --epochs $EPOCHS --lr $LR --lri $LRI --test_set_id 5 --nodes_num $NODES_NUM --num_classes $NUM_CLASSES --atten_type $ATTEN_TYPE --fusion True --batch_size $BATCH_SIZE  --dataset_name $DATASET_NAME --pretrain >> $LOG_FILE5 &

#if [[ $DATASET_NAME = 'US8K' ]];
#then
#echo 'empty'
#
#LOG_FILE6=""$LOG_DIR""$current_time""$MODEL_TYPE""_""$DATASET_NAME""_""$BATCH_SIZE""_""$NODES_NUM""_6.txt""
#LOG_FILE7=""$LOG_DIR""$current_time""$MODEL_TYPE""_""$DATASET_NAME""_""$BATCH_SIZE""_""$NODES_NUM""_7.txt""
#LOG_FILE8=""$LOG_DIR""$current_time""$MODEL_TYPE""_""$DATASET_NAME""_""$BATCH_SIZE""_""$NODES_NUM""_8.txt""
#LOG_FILE9=""$LOG_DIR""$current_time""$MODEL_TYPE""_""$DATASET_NAME""_""$BATCH_SIZE""_""$NODES_NUM""_9.txt""
#LOG_FILE10=""$LOG_DIR""$current_time""$MODEL_TYPE""_""$DATASET_NAME""_""$BATCH_SIZE""_""$NODES_NUM""_10.txt""

#echo $LOG_FILE6,$LOG_FILE7,$LOG_FILE8,$LOG_FILE9,$LOG_FILE10

#CUDA_VISIBLE_DEVICES=5 python audio_gacn_med.py --model_type $MODEL_TYPE --epochs $EPOCHS --lr $LR --lri $LRI --test_set_id 6 --nodes_num $NODES_NUM --fusion True --bs $BATCH_SIZE  --dataset_name $DATASET_NAME >> $LOG_FILE6 &
#CUDA_VISIBLE_DEVICES=6 python audio_gacn_med.py --model_type $MODEL_TYPE --epochs $EPOCHS --lr $LR --lri $LRI --test_set_id 7 --nodes_num $NODES_NUM --fusion True --bs $BATCH_SIZE  --dataset_name $DATASET_NAME >> $LOG_FILE7 &
#CUDA_VISIBLE_DEVICES=7 python audio_gacn_med.py --model_type $MODEL_TYPE --epochs $EPOCHS --lr $LR --lri $LRI --test_set_id 8 --nodes_num $NODES_NUM --fusion True --bs $BATCH_SIZE  --dataset_name $DATASET_NAME >> $LOG_FILE8 &
#CUDA_VISIBLE_DEVICES=8 python audio_gacn_med.py --model_type $MODEL_TYPE --epochs $EPOCHS --lr $LR --lri $LRI --test_set_id 9 --nodes_num $NODES_NUM --fusion True --bs $BATCH_SIZE  --dataset_name $DATASET_NAME >> $LOG_FILE9 &
#CUDA_VISIBLE_DEVICES=9 python audio_gacn_med.py --model_type $MODEL_TYPE --epochs $EPOCHS --lr $LR --lri $LRI --test_set_id 10 --nodes_num $NODES_NUM --fusion True --bs $BATCH_SIZE  --dataset_name $DATASET_NAME >> $LOG_FILE10 &

#fi
