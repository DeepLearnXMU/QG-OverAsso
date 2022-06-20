#!/usr/bin/env bash

DATA_DIR='../saved_data/data_en'
OUTPUT_DIR='../saved_data'

cd transformers
CUDA_VISIBLE_DEVICES=0 python examples/pytorch/wizard_of_internet/run.py \
    --model_name_or_path google/t5-v1_1-base \
    --do_train \
    --do_predict \
    --train_file "$DATA_DIR/train.json" \
    --validation_file "$DATA_DIR/valid.json" \
    --test_file "$DATA_DIR/test.json" \
    --source_prefix "" \
    --output_dir "$OUTPUT_DIR/t5-v1_1-base" \
    --per_device_train_batch_size=8 \
    --per_device_eval_batch_size=8 \
    --gradient_accumulation_steps 8 \
    --predict_with_generate \
    --num_train_epochs 12 \
    --text_column="dialogue" \
    --summary_column="query" \
    --remove_unused_columns false \
    --save_steps=1000 \
    --max_source_length 1024 \
    --max_target_length 128 \
    --num_beams 4 \
    --generation_num_beams 4 \
    --overwrite_output_dir