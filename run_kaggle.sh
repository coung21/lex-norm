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
else
    # BARTpho training
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
