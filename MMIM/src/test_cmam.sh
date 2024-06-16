#!/bin/bash
CMAM_PATH="results/saved_models/mosi/cmam_f1.pt"
DATASET="mosi"
echo "Evaluating $DATASET dataset"
i=1

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
--test_changed_pct 1.0 \
--is_test \
--save_model_to "results/saved_models/$DATASET/baseline_$i.pt" \
--save_results_to "results/$DATASET/test_baseline_missing_cmam_${i}_1.0.csv" \
--cmam_path $CMAM_PATH

