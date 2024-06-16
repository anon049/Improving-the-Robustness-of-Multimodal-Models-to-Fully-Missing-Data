#!/bin/bash
MODELS_PATH="results/saved_models"
RESULTS_PATH="results"
DATASET="mosi"



for i in {1..3}; do
    
    python src/extract_encoders.py\
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
    --save_results_to $RESULTS_PATH/$DATASET/baseline_$i.csv \
    --audio_encoder_path $MODELS_PATH/$DATASET/audio_encoder_$i.pt \
    --video_encoder_path $MODELS_PATH/$DATASET/video_encoder_$i.pt \
    
    if [ $? -eq 0 ]; then
        echo "mosi baseline $i done"
    else
        echo "mosi baseline $i failed"
        exit 1
    fi
    
done