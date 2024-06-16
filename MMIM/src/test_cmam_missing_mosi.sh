#!/bin/bash
# CMAM_PATH="/home/jmg/code/MMIM/mosi_cmam_models/mosi_cmam_0.35133"
CMAM_PATH="results/saved_models/mosi/cmam_f1.pt"
DATASET="mosi"

echo "Evaluating $DATASET dataset"

PCTS=(0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)

# PCTS=(0.9 1.0)

for pct in ${PCTS[@]}; do
    for i in {1..3}; do
        
        python src/test.py \
        --dataset $DATASET \
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
        --test_changed_pct $pct \
        --is_test \
        --save_model_to "results/saved_models/$DATASET/baseline_$i.pt" \
        --save_results_to ../results/mosi/missing/$DATASET/baseline_missing_w_cmam_${pct}.csv \
        --cmam_path $CMAM_PATH
        
        if [ $? -ne 0 ]; then
            echo "Error in training"
            exit 1
        fi
    done
done
