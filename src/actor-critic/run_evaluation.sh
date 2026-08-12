#!/bin/bash
set -e

# Cleanup function to handle script termination
cleanup() {
    # Kill any remaining python processes
    pkill -f "python.*vllm_infer.py" || true
    # Wait a moment for processes to clean up
    sleep 1
}

# Set up trap for script termination
trap cleanup EXIT

# Default values
ROOT_DIR="/teamspace/studios/this_studio"
LEVEL=1
MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
TEMPLATE="llama3"
TEMPERATURE=0.2
TOP_P=0.5
WORKFLOW="actor-critic" # Can be 'actor-critic' or 'self-reflection'
QUANTIZATION="" # Can be 'bitsandbytes', 'gptq', 'awq', etc. or empty for no quantization

# Calculate the experiment counter for this model and level
get_next_counter() {
    local model_name=$(basename "$MODEL_NAME")
    local step1_dir="$ROOT_DIR/Minds-in-the-Machine/results/actor_critic_results/${model_name}/level-${LEVEL}/step1"
    local file_count=0
    
    if [ -d "$step1_dir" ]; then
        # Count only .jsonl files in step1 directory, silently ignoring errors if directory is empty
        file_count=$(find "$step1_dir" -maxdepth 1 -name "*.jsonl" -type f 2>/dev/null | wc -l)
    fi

    # Format file count with leading zeros
    printf "%03d" $((file_count + 1))
}

# Load Hugging Face token from config.json
CONFIG_FILE="$ROOT_DIR/Minds-in-the-Machine/configs.json"

if [ -f "$CONFIG_FILE" ]; then
    HF_TOKEN=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['HF_TOKEN'])")
    if [[ "$HF_TOKEN" == "null" || -z "$HF_TOKEN" ]]; then
        echo "Error: HF_TOKEN is missing or null in configs.json"
        exit 1
    else
        echo "HuggingFace token set successfully: $HF_TOKEN"
    fi
else
    echo "Error: configs.json not found at $CONFIG_FILE"
    exit 1
fi

huggingface-cli login --token "$HF_TOKEN"

# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --level) LEVEL="$2"; shift ;;
        --model_name) MODEL_NAME="$2"; shift ;;
        --template) TEMPLATE="$2"; shift ;;
        --temperature) TEMPERATURE="$2"; shift ;;
        --top_p) TOP_P="$2"; shift ;;
        --workflow) WORKFLOW="$2"; shift ;;
        --quantization) QUANTIZATION="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Calculate counter value after parsing arguments to use correct LEVEL
COUNTER=$(get_next_counter)

# Create necessary directories with correct LEVEL value
model_name=$(basename "$MODEL_NAME")
mkdir -p "$ROOT_DIR/Minds-in-the-Machine/results/actor_critic_results/${model_name}/level-${LEVEL}"/{step1,step2,step2-parsed,step3}

echo "Using experiment counter: $COUNTER"
echo "Starting evaluation for Level: $LEVEL, Model: $MODEL_NAME, Workflow: $WORKFLOW"
if [[ -n "$QUANTIZATION" ]]; then
    echo "Using quantization: $QUANTIZATION"
else
    echo "No quantization specified (using full precision)"
fi

# Define project directories
MIM_DIR="Minds-in-the-Machine"
LLAMA_FACTORY_DIR="LLaMA-Factory"

# Helper function to run a command in a specific directory
run_in_dir() {
    local dir="$1"
    shift
    echo "==> Running in '$dir': $@"
    (
        cd "$dir" && "$@"
    )
}

cd "$ROOT_DIR"
cp ./$MIM_DIR/scripts/dataset_info.json ./$LLAMA_FACTORY_DIR/data/
cp ./$MIM_DIR/scripts/vllm_infer.py     ./$LLAMA_FACTORY_DIR/scripts/vllm_infer.py
mkdir -p ./LLaMA-Factory/data/tom_eval/step{1,2,3}

# Step 1: Format dataset and run initial inference (Common for both workflows)
echo "--- Step 1: Initial Actor Inference ---"
run_in_dir "$MIM_DIR" python scripts/format_alpaca_dataset.py --step 1 --level "$LEVEL"
run_in_dir "$LLAMA_FACTORY_DIR" python scripts/vllm_infer.py --step 1 --level "$LEVEL" --model_name_or_path "$MODEL_NAME" --template "$TEMPLATE"  --temperature "$TEMPERATURE" --top_p "$TOP_P" --quantization "$QUANTIZATION" --counter "$COUNTER"

if [[ "$WORKFLOW" == "actor-critic" ]]; then
    echo "--- Running Actor-Critic Workflow ---"
    
    # Step 2: Process results and run critic
    echo "--- Step 2: Processing and Critic ---"
    run_in_dir "$MIM_DIR" python scripts/format_alpaca_dataset.py --step 2 --level "$LEVEL" --model_name "$MODEL_NAME" --counter "$COUNTER"
    run_in_dir "$LLAMA_FACTORY_DIR" python scripts/vllm_infer.py --step 2 --level "$LEVEL" --model_name_or_path "$MODEL_NAME" --template "$TEMPLATE" --quantization "$QUANTIZATION" --counter "$COUNTER"
    run_in_dir "$MIM_DIR" python scripts/parse_predicts.py --level "$LEVEL" --model_name "$MODEL_NAME" --counter "$COUNTER"

    # Step 3: Actor Revision
    echo "--- Step 3: Actor Revision ---"
    run_in_dir "$MIM_DIR" python scripts/create_revision_dataset.py --level "$LEVEL" --model_name "$MODEL_NAME" --counter "$COUNTER"
    run_in_dir "$LLAMA_FACTORY_DIR" python scripts/vllm_infer.py --step 3 --level "$LEVEL" --model_name_or_path "$MODEL_NAME" --template "$TEMPLATE"  --temperature "$TEMPERATURE" --top_p "$TOP_P" --quantization "$QUANTIZATION" --counter "$COUNTER"

elif [[ "$WORKFLOW" == "self-reflection" ]]; then
    echo "--- Running Self-Reflection Workflow ---"
    
    # Step 2: Create self-reflection dataset and run revision
    echo "--- Step 2: Self-Reflection and Revision ---"
    run_in_dir "$MIM_DIR" python scripts/create_self_reflection_dataset.py --level "$LEVEL" --model_name "$MODEL_NAME" --counter "$COUNTER"
    run_in_dir "$LLAMA_FACTORY_DIR" python scripts/vllm_infer.py --step 2 --level "$LEVEL" --model_name_or_path "$MODEL_NAME" --template "$TEMPLATE" --quantization "$QUANTIZATION" --counter "$COUNTER"

else
    echo "Unknown workflow: $WORKFLOW. Please choose 'actor-critic' or 'self-reflection'."
    exit 1
fi

echo "Workflow completed successfully for Level: $LEVEL, Model: $MODEL_NAME"
if [[ -n "$QUANTIZATION" ]]; then
    echo "Quantization used: $QUANTIZATION"
fi

# Print experiment details for reference
echo "Experiment details:"
echo "  Counter: $COUNTER"
echo "  Level: $LEVEL"
echo "  Model: $MODEL_NAME"
echo "  Workflow: $WORKFLOW"

exit 0