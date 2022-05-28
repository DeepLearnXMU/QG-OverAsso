#!/usr/bin/env bash

for step in 50 100 150 200 250 300
do
  CUDA_VISIBLE_DEVICES=2 python main_ra.py "test" "en" ${step}
done
