#!/bin/bash
# Git initialization and push script for AlphaRL-Quant

echo "=========================================="
echo "AlphaRL-Quant: GitHub Repository Setup"
echo "=========================================="
echo ""

# Step 1: Initialize repository
if [ ! -d ".git" ]; then
    echo "[1/5] Initializing git repository..."
    git init
    echo "      Repository initialized"
else
    echo "[1/5] Git repository already initialized"
fi
echo ""

# Step 2: Add all files
echo "[2/5] Adding files to git..."
git add .
echo "      Files staged for commit"
echo ""

# Step 3: Create initial commit
echo "[3/5] Creating initial commit..."
if [ -z "$(git log --oneline 2>/dev/null)" ]; then
    git commit -m "Initial commit: Production-grade RL trading system

- Centralized configuration management
- Professional logging infrastructure
- Type-safe codebase with defensive assertions
- Comprehensive documentation suite
- Backtest results: Sharpe 1.29, +53.51% returns"
    echo "      Initial commit created"
else
    echo "      Commits already exist, skipping initial commit"
fi
echo ""

# Step 4: Add remote (if not exists)
echo "[4/5] Adding remote repository..."
if ! git remote | grep -q "origin"; then
    git remote add origin https://github.com/Rushabh333/AlphaRL-Quant.git
    echo "      Remote 'origin' added"
else
    echo "      Remote 'origin' already exists"
fi
echo ""

# Step 5: Push to GitHub
echo "[5/5] Ready to push to GitHub"
echo ""
echo "Run the following command to push:"
echo ""
echo "    git push -u origin main"
echo ""
echo "Or if using master branch:"
echo ""
echo "    git branch -M main"
echo "    git push -u origin main"
echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
