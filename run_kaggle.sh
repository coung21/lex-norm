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

# Get experiment from argument, default to bartpho if not provided
EXPERIMENT=${1:-bartpho}

echo ""
echo "====================================="
echo "   Starting Training: $EXPERIMENT    "
echo "====================================="

if [ "$EXPERIMENT" == "rule_based" ]; then
    # Rule-based baseline (no training needed)
    python rule_based_baseline.py \
        --train data/ViLexNorm/data/train.csv \
        --test data/ViLexNorm/data/test.csv \
        --dev data/ViLexNorm/data/dev.csv \
        --output outputs/rule_based
elif [ "$EXPERIMENT" == "augmented" ]; then
    # Augmented BARTpho training
    AUG_DATA="data/pseudo_label/train_augmented.csv"
    
    # 1. Search locally first
    if [ ! -f "$AUG_DATA" ]; then
        echo "Local augmented data not found. Searching in /kaggle/input..."
        
        # 2. Search in all Kaggle Input datasets
        SEARCH_RESULT=$(find /kaggle/input -name "train_augmented.csv" -print -quit 2>/dev/null)
        
        if [ -n "$SEARCH_RESULT" ]; then
            echo "Found augmented data at: $SEARCH_RESULT"
            # Ensure local directory exists
            mkdir -p data/pseudo_label
            # Link or copy to local path for consistency
            cp "$SEARCH_RESULT" "$AUG_DATA"
        else
            echo "ERROR: Could not find train_augmented.csv locally or in /kaggle/input."
            echo "Please upload the file as a Kaggle Dataset before running."
            exit 1
        fi
    fi
    
    python train.py \
        --config config.yaml \
        --train_csv "$AUG_DATA" \
        --experiment bartpho-augmented

    python evaluate.py \
        --checkpoint outputs/bartpho/best_model \
        --config config.yaml \
        --split test dev \
        --experiment bartpho-augmented
else
    # Standard BARTpho training
    python train.py \
        --config config.yaml \
        --experiment $EXPERIMENT

    echo ""
    echo "====================================="
    echo "   Starting Evaluation               "
    echo "====================================="

    python evaluate.py \
        --checkpoint outputs/bartpho/best_model \
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
