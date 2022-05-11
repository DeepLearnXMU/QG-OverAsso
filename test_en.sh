#!/usr/bin/env bash

for step in 1000 2000 3000 4000 5000 6000 7000 8000 9000
do
  DATA_DIR='/data/home/antewang/query-gen/saved_data/woi_data_ext_gen_v2'
  MODEL_DIR=/data/home/antewang/query-gen/saved_data/t5-v1_1-base-ext-gen-v2/checkpoint-${step}

  cd transformers
  CUDA_VISIBLE_DEVICES=1 python examples/pytorch/summarization/run.py \
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
      --max_target_length 50 \
      --num_beams 4 \
      --generation_num_beams 4 \
      --predict_output_file "generated_predictions.txt"
done