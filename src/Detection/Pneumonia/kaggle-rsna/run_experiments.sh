#!/usr/bin/env bash
CUR_DIR=$pwd

# Download dataset
# bash download_dataset.sh

# Run model training in debug mode
CUDA_VISIBLE_DEVICES=0 python src/train_runner.py --action train --debug True --train_csv ../../../detect_3_class_train.csv --val_csv ../../../detect_3_class_val.csv

# # Run full model training
# CUDA_VISIBLE_DEVICES=0 python src/train_runner.py --action train --model resnet101_320 --debug False --num-epochs 16

# # Test model
# CUDA_VISIBLE_DEVICES=0 python src/train_runner.py --action test_model --model resnet101_320 --debug True --epoch 12

# check metrics
CUDA_VISIBLE_DEVICES=0 python src/train_runner.py --action check_metric --test_csv ../../../detect_3_class_test.csv

# Generate and save oof predictions
# CUDA_VISIBLE_DEVICES=0 python src/train_runner.py --action generate_predictions --model resnet101_320 --debug False --num-epochs 16

