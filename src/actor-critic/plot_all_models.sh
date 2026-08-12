#!/bin/bash

# Script to plot results for all models in results/actor_critic_results
RESULTS_DIR="../../results/actor_critic_results"
PLOT_SCRIPT="plot_overal_results.py"

for MODEL_DIR in "$RESULTS_DIR"/*/; do
    MODEL_NAME=$(basename "$MODEL_DIR")
    echo "Plotting for model: $MODEL_NAME"
    python "$PLOT_SCRIPT" --model_name "$MODEL_NAME" --results_dir "$RESULTS_DIR"
done 