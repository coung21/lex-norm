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

# Get experiment from argument, default to baseline if not provided
EXPERIMENT=${1:-baseline}

echo ""
echo "====================================="
echo "   Starting Training: $EXPERIMENT    "
echo "====================================="

# If you want to use wandb without prompting, make sure to add WANDB_API_KEY 
# to Kaggle Secrets and run: export WANDB_API_KEY=your_key_here before running this script
python train.py --experiment $EXPERIMENT --epochs 4

echo ""
echo "====================================="
echo "   Starting Evaluation               "
echo "====================================="

if [ "$EXPERIMENT" == "baseline" ]; then
    python evaluate.py --checkpoint outputs/baseline/best_model.pt --experiment baseline
else
    # For contrastive, we evaluate the final stage2 decoder
    python evaluate.py --checkpoint outputs/contrastive/stage2/best_model.pt --experiment contrastive
fi

echo ""
echo "====================================="
echo "   Run completed successfully!       "
echo "====================================="
