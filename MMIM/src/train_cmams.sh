#!/bin/bash
MODELS_PATH="results/saved_models"
RESULTS_PATH="results"
DATASET="mosi"

mkdir -p results/$DATASET/missing
mkdir -p results/saved_models/$DATASET/cmams

for i in {1..3}; do
    python src/run_cmam.py \
    --dataset $DATASET \
    --contrast \
    --d_vh 32 \
    --d_ah 32 \
    --test_method missing \
    --test_changed_modal language \
    --test_changed_pct 1.0 \
    --baseline_model_path $MODELS_PATH/$DATASET/baseline_$i.pt \
    --save_results_to $RESULTS_PATH/$DATASET/baseline_cmam_train.csv \
    --cmam_path $MODELS_PATH/$DATASET/$i/cmam.pt \
    --audio_encoder_path $MODELS_PATH/$DATASET/audio_encoder_$i.pt \
    --video_encoder_path $MODELS_PATH/$DATASET/video_encoder_$i.pt
    
    if [ $? -eq 0 ]; then
        echo "mosi baseline $i done"
    else
        echo "mosi baseline $i failed"
        exit 1
    fi
done