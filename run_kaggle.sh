#!/bin/bash

# Exit on error
set -e

echo "====================================="
echo "   Setting up Lex-Norm on Kaggle     "
echo "====================================="

# Init and pull git submodules (to get data/ViLexNorm dataset)
echo "Pulling git submodules..."
git submodule update --init --recursive

# Install dependencies (Kaggle usually has torch, we install the rest)
echo "Installing requirements..."
pip install -r requirements.txt

# --- Wandb setup ---
# On Kaggle, add WANDB_API_KEY to Kaggle Secrets, then:
#   from kaggle_secrets import UserSecretsClient
#   secrets = UserSecretsClient()
#   os.environ["WANDB_API_KEY"] = secrets.get_secret("WANDB_API_KEY")
# Or set it manually before running this script:
#   export WANDB_API_KEY=your_key_here
if [ -z "$WANDB_API_KEY" ]; then
    echo "WARNING: WANDB_API_KEY not set. wandb will run in offline mode."
    export WANDB_MODE=offline
fi

# Get experiment from argument, default to byt5-small if not provided
EXPERIMENT=${1:-byt5-small}

echo ""
echo "====================================="
echo "   Starting Training: $EXPERIMENT    "
echo "====================================="

# Helper function to find and setup data from Kaggle Input
find_and_setup_data() {
    local FILE_NAME=$1
    local TARGET_PATH=$2
    
    if [ ! -f "$TARGET_PATH" ]; then
        echo "Local file $TARGET_PATH not found. Searching in /kaggle/input..."
        local SEARCH_RESULT=$(find /kaggle/input -name "$FILE_NAME" -print -quit 2>/dev/null)
        
        if [ -n "$SEARCH_RESULT" ]; then
            echo "Found file at: $SEARCH_RESULT"
            mkdir -p "$(dirname "$TARGET_PATH")"
            cp "$SEARCH_RESULT" "$TARGET_PATH"
        else
            echo "ERROR: Could not find $FILE_NAME locally or in /kaggle/input."
            exit 1
        fi
    fi
}

if [ "$EXPERIMENT" == "rule_based" ]; then
    # Rule-based baseline
    python rule_based_baseline.py \
        --train data/ViLexNorm/data/train.csv \
        --test data/ViLexNorm/data/test.csv \
        --dev data/ViLexNorm/data/dev.csv \
        --output outputs/rule_based
elif [ "$EXPERIMENT" == "augmented" ]; then
    # Augmented training
    AUG_DATA="data/pseudo_label/train_augmented.csv"
    find_and_setup_data "train_augmented.csv" "$AUG_DATA"
    
    python train.py \
        --config config.yaml \
        --train_csv "$AUG_DATA" \
        --experiment byt5-augmented
    
    OUTPUT_DIR=$(grep "output_dir" config.yaml | cut -d'"' -f2)
    python evaluate.py \
        --checkpoint "${OUTPUT_DIR:-outputs/byt5-small}/best_model" \
        --config config.yaml \
        --split test dev \
        --experiment byt5-augmented
elif [ "$EXPERIMENT" == "filtered" ]; then
    # Filtered Augmented training
    FILTERED_DATA="data/pseudo_label/train_filtered.csv"
    find_and_setup_data "train_filtered.csv" "$FILTERED_DATA"
    
    python train.py \
        --config config.yaml \
        --train_csv "$FILTERED_DATA" \
        --experiment byt5-filtered
    
    OUTPUT_DIR=$(grep "output_dir" config.yaml | cut -d'"' -f2)
    python evaluate.py \
        --checkpoint "${OUTPUT_DIR:-outputs/byt5-small}/best_model" \
        --config config.yaml \
        --split test dev \
        --experiment byt5-filtered
else
    # Standard training
    python train.py \
        --config config.yaml \
        --experiment $EXPERIMENT

    echo ""
    echo "====================================="
    echo "   Starting Evaluation               "
    echo "====================================="

    OUTPUT_DIR=$(grep "output_dir" config.yaml | cut -d'"' -f2)
    python evaluate.py \
        --checkpoint "${OUTPUT_DIR:-outputs/byt5-small}/best_model" \
        --config config.yaml \
        --split test dev \
        --experiment $EXPERIMENT
fi

# Copy outputs to /kaggle/working/ for download (if on Kaggle)
if [ -d "/kaggle/working" ]; then
    echo ""
    echo "Copying outputs to /kaggle/working/..."
    cp -r outputs/ /kaggle/working/
fi

echo ""
echo "====================================="
echo "   Run completed successfully!       "
echo "====================================="
