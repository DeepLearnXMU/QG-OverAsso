#!/usr/bin/env bash

DATA_DIR='../saved_data/woi_data_np'
MODEL_DIR=../model_zh/

cd transformers
CUDA_VISIBLE_DEVICES=4 python examples/pytorch/wizard_of_internet/service.py \
    --model_name_or_path $MODEL_DIR \
    --do_predict \
    --train_file "$DATA_DIR/train.json" \
    --validation_file "$DATA_DIR/valid.json" \
    --test_file "$DATA_DIR/valid.json" \
    --source_prefix "" \
    --output_dir "$MODEL_DIR" \
    --per_device_train_batch_size=16 \
    --per_device_eval_batch_size=16 \
    --predict_with_generate \
    --learning_rate 3e-5 \
    --num_train_epochs 30 \
    --text_column="dialogue" \
    --summary_column="query" \
    --save_steps=30000 \
    --max_target_length 64 \
    --num_beams 4 
