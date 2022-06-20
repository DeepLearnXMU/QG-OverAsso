#!/usr/bin/env bash

DATA_DIR='../saved_data/data_zh'
OUTPUT_DIR='../saved_data'
STOPWORDS_PATH='examples/pytorch/wizard_of_internet/cn_stopwords.txt'

cd transformers
CUDA_VISIBLE_DEVICES=0 python examples/pytorch/wizard_of_internet/run.py \
    --model_name_or_path Langboat/mengzi-t5-base \
    --do_train \
    --do_predict \
    --train_file "$DATA_DIR/train.json" \
    --validation_file "$DATA_DIR/dev.json" \
    --test_file "$DATA_DIR/dev.json" \
    --source_prefix "" \
    --output_dir "$OUTPUT_DIR/mengzi-t5-base" \
    --per_device_train_batch_size=16 \
    --per_device_eval_batch_size=16 \
    --gradient_accumulation_steps 4 \
    --predict_with_generate \
    --num_train_epochs 40 \
    --text_column="dialogue" \
    --summary_column="query" \
    --remove_unused_columns false \
    --save_steps=1000 \
    --max_source_length 1024 \
    --max_target_length 128 \
    --num_beams 4 \
    --generation_num_beams 4 \
    --overwrite_output_dir
