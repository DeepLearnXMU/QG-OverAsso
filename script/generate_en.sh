#!/usr/bin/env bash

for i in 0 1 2
do
    DATA_DIR='../saved_data/woi_data_3f'
    MODEL_DIR=../saved_data/t5-v1_1-base-${i}f

    cd transformers
    CUDA_VISIBLE_DEVICES=3 python examples/pytorch/wizard_of_internet/run_bm.py \
      --model_name_or_path $MODEL_DIR \
      --do_predict \
      --train_file "$DATA_DIR/valid.json" \
      --validation_file "$DATA_DIR/valid.json" \
      --test_file "$DATA_DIR/train_${i}_.json" \
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
      --max_target_length 50 \
      --num_beams 10 \
      --generation_num_beams 10 \
      --predict_output_file "bm_${i}_generated_predictions.txt"
done