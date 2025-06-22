#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status

# Wandb login
WANDB_API_KEY="xxxxxxxxxxxxxx"  # Replace with your actual API key
echo "Logging in to Weights & Biases..."
wandb login $WANDB_API_KEY

# Dataset configuration
DATASETS=(
    # "037_WCE"       # multiclass
    "038_HAM10000"  # multiclass
    "039_RFMiD"     # multilabel
    "044_BUSI"      # binary
    # Add more datasets here
)

WANDB_ENTITY="yxxxxxxxxxxxxxx"  # Updated to match configuration

# Function to run training with specific model
run_training() {
    local model=$1
    local size=$2
    local env=$3
    local dataset_name=$4
    shift 4  # Remove the first 5 arguments
    local extra_args="$@"  # Capture any additional arguments
    
    echo "Training $model-$size on $dataset_name..."

    conda run -n $env python /home/sunanhe/luoyi/model_eval/scripts/train_${model}.py \
        --dataset_name $dataset_name \
        --model_name $model \
        --model_size $size \
        --wandb_entity $WANDB_ENTITY \
        --wandb_project "medical-image-classification" \
        --wandb_name "${model}_${size}_${dataset_name}_$(date +%Y%m%d_%H%M%S)" \
        $extra_args
}

# Experimental parameters
learning_rates=(0.002)
weight_decay=0.01
epochs=100
device="cuda:2"

# Train on each dataset
for dataset_name in "${DATASETS[@]}"; do
    echo "Starting training on dataset: $dataset_name"

    # Train MedViT models (size: base, large)
    echo "Training MedViT models..."
    for size in base large; do
        for lr in "${learning_rates[@]}"; do
            echo "Training with lr=$lr, weight_decay=$weight_decay, model_size=$size"
            
            # Set batch size based on model size
            if [ "$size" == "small" ]; then
                batch_size=48
            elif [ "$size" == "base" ]; then
                batch_size=32
            elif [ "$size" == "large" ]; then
                batch_size=32
            else
                echo "Unknown model size: $size"
                continue
            fi

            run_training "medvit" $size "medvitv2-ly" $dataset_name \
                --batch_size $batch_size \
                --device $device \
                --learning_rate $lr \
                --weight_decay $weight_decay \
                --epochs $epochs
        done
    done

    echo "Completed training on dataset: $dataset_name"
    echo "----------------------------------------"
done

echo "All training completed!"