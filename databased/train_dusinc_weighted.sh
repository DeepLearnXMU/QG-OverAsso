#!/usr/bin/env bash

DATA_DIR='../saved_data/data_dusinc'
OUTPUT_DIR='../saved_data'
exponent=0.5

CUDA_VISIBLE_DEVICES=5 python run_model_weighted.py \
    --model_name_or_path Langboat/mengzi-t5-base \
    --do_train \
    --do_predict \
    --train_file "$DATA_DIR/train.json" \
    --validation_file "$DATA_DIR/dev.json" \
    --test_file "$DATA_DIR/dev.json" \
    --source_prefix "" \
    --output_dir "$OUTPUT_DIR/mengzi-t5-base-dusinc_exponent-${exponent}" \
    --per_device_train_batch_size=16 \
    --per_device_eval_batch_size=16 \
    --gradient_accumulation_steps 1 \
    --predict_with_generate \
    --num_train_epochs 10 \
    --save_total_limit 5 \
    --save_steps 500 \
    --save_strategy steps \
    --eval_steps 500 \
    --evaluation_strategy steps \
    --warmup_steps 500 \
    --weight_decay 0.005 \
    --text_column="dialogue" \
    --summary_column="query" \
    --remove_unused_columns false \
    --max_source_length 400 \
    --max_target_length 50 \
    --lang zh \
    --num_beams 4 \
    --seed 200 \
    --generation_num_beams 4 \
    --exponent ${exponent} \
    --overwrite_output_dir