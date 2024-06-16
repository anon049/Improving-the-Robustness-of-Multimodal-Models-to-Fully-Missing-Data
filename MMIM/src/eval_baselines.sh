#!/bin/bash

MODELS_PATH="results/saved_models"
RESULTS_PATH="results"
DATASET="mosi"

mkdir -p results/$DATASET/missing
mkdir -p results/saved_models/$DATASET/cmams


for i in {1..3}; do
    
    python src/test.py \
    --dataset mosi \
    --contrast \
    --lr_main 1e-3 \
    --lr_mmilb 4e-3 \
    --alpha 0.3 \
    --beta 0.1 \
    --batch_size 32 \
    --d_vh 32 \
    --d_ah 32 \
    --train_method hybird \
    --train_changed_modal language \
    --train_changed_pct 0.0 \
    --test_method missing \
    --test_changed_modal language \
    --test_changed_pct 0.0 \
    --save_model_to $MODELS_PATH/$DATASET/baseline_$i.pt \
    --save_results_to $RESULTS_PATH/$DATASET/baseline_$i.csv
    
    if [ $? -eq 0 ]; then
        echo "mosi baseline $i done"
    else
        echo "mosi baseline $i failed"
        exit 1
    fi
    
done