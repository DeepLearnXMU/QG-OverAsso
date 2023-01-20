#!/usr/bin/env bash

DATA_DIR='../saved_data/data_woi'
OUTPUT_DIR='../saved_data'

CUDA_VISIBLE_DEVICES=5 python run_model.py \
    --model_name_or_path facebook/bart-base \
    --do_train \
    --do_predict \
    --train_file "$DATA_DIR/train.json" \
    --validation_file "$DATA_DIR/valid.json" \
    --test_file "$DATA_DIR/test.json" \
    --source_prefix "" \
    --output_dir "$OUTPUT_DIR/bart-base-woi" \
    --per_device_train_batch_size=16 \
    --per_device_eval_batch_size=16 \
    --gradient_accumulation_steps 4 \
    --predict_with_generate \
    --num_train_epochs 10 \
    --save_strategy epoch \
    --save_total_limit 5 \
    --evaluation_strategy epoch \
    --warmup_steps 1000 \
    --weight_decay 0.005 \
    --text_column="dialogue" \
    --summary_column="query" \
    --remove_unused_columns false \
    --max_source_length 650 \
    --max_target_length 100 \
    --lang en \
    --num_beams 4 \
    --generation_num_beams 4 \
    --overwrite_output_dir