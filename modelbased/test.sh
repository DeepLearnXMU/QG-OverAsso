#!/usr/bin/env bash

step=400
CUDA_VISIBLE_DEVICES=2 python main_ra.py "test" "zh" ${step}
