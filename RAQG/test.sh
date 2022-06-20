#!/usr/bin/env bash

step=100 # e.g. checkpoint-100
CUDA_VISIBLE_DEVICES=0 python main_ra.py "test" "en" ${step}

# for dusinc
# CUDA_VISIBLE_DEVICES=0 python main_ra.py "test" "zh" ${step}

