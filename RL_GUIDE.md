# RL Trading Agent - Quick Start Guide

##  Training the Agent

### Quick Test (10K steps - ~10 minutes)
```bash
python src/training/train_agent.py
```

### Full Training (1M steps - ~2-4 hours)
Edit `src/training/train_agent.py` line 134:
```python
model, test_data = train_trading_agent(total_timesteps=1_000_000)
```

##  Monitor Training

### Launch TensorBoard
```bash
tensorboard --logdir=./logs/tensorboard/
```
Open: http://localhost:6006

### Watch Key Metrics
- **ep_rew_mean**: Should trend upward
- **policy_loss**: Should decrease
- **value_loss**: Should stabilize

##  Evaluate Agent

### Run Backtest
```bash
python src/evaluation/backtest.py
```

### View Results
```bash
open reports/backtest_results.png  # Visualization
cat reports/portfolio_history.csv  # Portfolio values
cat reports/trades_history.csv     # All trades
```

##  Success Criteria

Good agent performance:
- Total Return > Buy-and-Hold
- Sharpe Ratio > 1.0
- Max Drawdown < 15%
- Win Rate > 50%

##  Troubleshooting

### Agent not learning?
- Check reward scale in `trading_env.py`
- Increase `ent_coef` in training script
- Try 'simple' reward instead of 'sharpe'

### Training crashes?
- Check for NaN in features
- Verify data has no missing values
- Reduce learning rate

##  File Structure

```
src/
 models/
    trading_env.py      # RL environment
 training/
    train_agent.py       # Training script  
 evaluation/
     backtest.py          # Backtesting

data/processed/
 features.csv             # Input data (3,091 rows × 39 features)

models/
 best/                    # Best model checkpoints
 checkpoints/             # Training checkpoints
 ppo_trading_final.zip   # Final trained model

logs/
 tensorboard/             # TensorBoard logs
 eval/                    # Evaluation logs

reports/
 backtest_results.png     # Performance visualization
 portfolio_history.csv    # Portfolio values over time
 trades_history.csv       # Trade history
```

##  Next Steps

1. **Experiment with rewards**: Try 'simple', 'sharpe', 'sortino'
2. **Tune hyperparameters**: Learning rate, network size
3. **Feature selection**: Test with fewer features
4. **Multi-ticker**: Train on different stocks
5. **Live paper trading**: Test in simulation

##  Resources

- Stable-Baselines3 Docs: https://stable-baselines3.readthedocs.io/
- FinRL Library: https://github.com/AI4Finance-Foundation/FinRL
- RL Trading Papers: https://arxiv.org/abs/2011.09607
