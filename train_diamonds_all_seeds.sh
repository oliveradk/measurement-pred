#!/bin/bash

# Option 1: Using seq
seeds=$(seq 1 7)
# # OR Option 2: Explicit list
# seeds="0 1 2 3 4 5 6 7"

datasets=$(echo $seeds | tr ' ' '\n' | sed 's/^/redwoodresearch\/diamonds-seed/' | paste -sd,)

# echo $datasets
python train.py --multirun model.dataset_name=$datasets