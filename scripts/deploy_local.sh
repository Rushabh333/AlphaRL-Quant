#!/bin/bash
# Local deployment script for AlphaRL-Quant
# Author: AlphaRL-Quant Team
# Purpose: Set up and run the system locally without Docker

set -e  # Exit on error

echo "🚀 AlphaRL-Quant Local Deployment"
echo "=================================="
echo ""

# =============================================================================
# 1. Check Python Version
# =============================================================================
echo "📋 Checking prerequisites..."

python_version=$(python3 --version 2>&1 | awk '{print $2}')
required_version="3.10"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "❌ Python 3.10+ required. Found: Python $python_version"
    echo "   Please install Python 3.10 or higher"
    exit 1
fi
echo "✅ Python version: $python_version"

# =============================================================================
# 2. Virtual Environment Setup
# =============================================================================
if [ ! -d ".venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv .venv
    echo "✅ Virtual environment created"
else
    echo "✅ Virtual environment already exists"
fi

# =============================================================================
# 3. Activate Virtual Environment and Install Dependencies
# =============================================================================
echo "📥 Installing dependencies..."
source .venv/bin/activate

# Upgrade pip
pip install -q --upgrade pip

# Install dependencies
pip install -q -r requirements.txt

# Install package in editable mode
pip install -q -e .

echo "✅ Dependencies installed"

# =============================================================================
# 4. Initialize Directory Structure
# =============================================================================
echo "📁 Creating directory structure..."
mkdir -p data/{raw,processed,cache}
mkdir -p logs
mkdir -p models/{best,checkpoints}
mkdir -p checkpoints
mkdir -p reports
mkdir -p config

# Create .gitkeep files
for dir in data/raw data/processed data/cache logs models/best models/checkpoints checkpoints reports; do
    touch $dir/.gitkeep
done

echo "✅ Directory structure ready"

# =============================================================================
# 5. Environment Configuration
# =============================================================================
if [ ! -f ".env" ]; then
    echo "⚠️  No .env file found. Creating from .env.example..."
    if [ -f ".env.example" ]; then
        cp .env.example .env
        echo "✅ .env file created"
        echo "⚠️  Please edit .env with your credentials before running the pipeline"
    else
        echo "⚠️  .env.example not found - skipping environment setup"
    fi
else
    echo "✅ .env file exists"
fi

# =============================================================================
# 6. Run Health Checks
# =============================================================================
echo ""
echo "🏥 Running health checks..."
if python scripts/health_check.py 2>&1 | grep -q "healthy"; then
    echo "✅ Health checks passed"
else
    echo "⚠️  Some health checks failed (non-critical, can proceed)"
fi

# =============================================================================
# 7. Run Data Pipeline
# =============================================================================
echo ""
echo "🔄 Running data pipeline..."
echo "This may take a few minutes depending on data size..."
echo ""

if python scripts/run_pipeline.py; then
    echo ""
    echo "✅ Pipeline completed successfully!"
else
    echo ""
    echo "❌ Pipeline failed. Check logs/pipeline.log for details"
    exit 1
fi

# =============================================================================
# 8. Summary
# =============================================================================
echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║                 ✅ DEPLOYMENT COMPLETE                     ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
echo "📊 Data Status:"
ls -lh data/processed/*.csv 2>/dev/null || echo "   No processed data files yet"
echo ""
echo "🎯 Next Steps:"
echo "   1. Train Model:"
echo "      python src/training/train_agent.py"
echo ""
echo "   2. Monitor Training (in another terminal):"
echo "      tensorboard --logdir=./logs/tensorboard/"
echo "      # Visit http://localhost:6006"
echo ""
echo "   3. View Logs:"
echo "      tail -f logs/pipeline.log"
echo ""
echo "   4. Run Backtest:"
echo "      python src/evaluation/backtest.py"
echo ""
echo "💡 Tip: Keep this virtual environment activated"
echo "   To reactivate: source .venv/bin/activate"
echo ""
