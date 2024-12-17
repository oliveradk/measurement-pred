#!/bin/bash

for seed in {0...7}; do
    python train.py --multirun model.dataset_name=redwoodresearch/diamonds-seed$seed &
    sleep 1
done