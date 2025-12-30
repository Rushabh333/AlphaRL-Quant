"""Backtesting and evaluation for trained RL agents."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from src.models.trading_env import TradingEnvironment
from src.utils.logger import get_logger

logger = get_logger(__name__)


def calculate_metrics(portfolio_values: list, trades: list) -> dict:
    """Calculate comprehensive trading metrics."""
    values = pd.Series(portfolio_values)
    returns = values.pct_change().dropna()
    
    total_return = (values.iloc[-1] / values.iloc[0] - 1) * 100
    
    # Risk metrics
    sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
    
    # Drawdown
    running_max = values.expanding().max()
    drawdown = (values - running_max) / running_max
    max_drawdown = drawdown.min() * 100
    
    # CAGR
    cagr = ((values.iloc[-1] / values.iloc[0]) ** (252 / len(values)) - 1) * 100
    
    # Calmar ratio
    calmar_ratio = abs(cagr / max_drawdown) if max_drawdown != 0 else 0
    
    return {
        'Total Return (%)': total_return,
        'CAGR (%)': cagr,
        'Sharpe Ratio': sharpe_ratio,
        'Max Drawdown (%)': max_drawdown,
        'Calmar Ratio': calmar_ratio,
        'Total Trades': len(trades),
        'Volatility (%)': returns.std() * np.sqrt(252) * 100
    }


def backtest_agent(model_path: str, test_data: pd.DataFrame, initial_capital: float = 10000.0):
    """Backtest trained RL agent."""
    logger.info("=" * 80)
    logger.info("BACKTESTING RL AGENT")
    logger.info("=" * 80)
    
    # Load model
    logger.info(f"Loading model from {model_path}...")
    model = PPO.load(model_path)
    
    logger.info("Creating test environment...")
    env = TradingEnvironment(
        data=test_data,
        initial_capital=initial_capital,
        transaction_cost=0.001,
        reward_type='simple',
        action_type='discrete'
    )
    
    # Run backtest
    logger.info("Running backtest...")
    obs, _ = env.reset()  # gymnasium returns (obs, info)
    done = False
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
    
    portfolio_history = env.get_portfolio_history()
    trades_history = env.get_trades_history()
    
    # Calculate metrics
    metrics = calculate_metrics(portfolio_history['portfolio_value'].tolist(), env.trades)
    
    # Buy & Hold baseline
    initial_price = test_data.iloc[env.window_size]['Close']
    final_price = test_data.iloc[-1]['Close']
    buy_hold_return = (final_price / initial_price - 1) * 100
    
    # Print results
    logger.info("\n" + "=" * 80)
    logger.info("BACKTEST RESULTS")
    logger.info("=" * 80)
    
    for metric, value in metrics.items():
        logger.info(f"{metric:25s}: {value:>10.2f}")
    
    logger.info("-" * 80)
    logger.info(f"{'Buy & Hold Return (%)':25s}: {buy_hold_return:>10.2f}")
    logger.info(f"{'Alpha (%)':25s}: {metrics['Total Return (%)'] - buy_hold_return:>10.2f}")
    logger.info("=" * 80)
    
    # Plot results
    plot_backtest_results(portfolio_history, test_data, trades_history, env.window_size)
    
    return metrics, portfolio_history, trades_history


def plot_backtest_results(portfolio_history, test_data, trades_history, window_size):
    """Create visualization of backtest results."""
    fig, axes = plt.subplots(2, 1, figsize=(15, 10))
    
    # Portfolio value over time
    axes[0].plot(portfolio_history['step'], portfolio_history['portfolio_value'])
    axes[0].set_title('Portfolio Value Over Time', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Step')
    axes[0].set_ylabel('Portfolio Value ($)')
    axes[0].grid(True, alpha=0.3)
    
    # Price chart with trades
    price_data = test_data.iloc[window_size:].reset_index(drop=True)
    axes[1].plot(price_data.index, price_data['Close'], label='Close Price')
    
    if not trades_history.empty:
        buy_trades = trades_history[trades_history['action'] == 'BUY']
        sell_trades = trades_history[trades_history['action'] == 'SELL']
        
        axes[1].scatter(buy_trades['step'] - window_size, buy_trades['price'],
                       color='green', marker='^', s=100, label='Buy', zorder=5)
        axes[1].scatter(sell_trades['step'] - window_size, sell_trades['price'],
                       color='red', marker='v', s=100, label='Sell', zorder=5)
    
    axes[1].set_title('Price Chart with Trading Signals', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Step')
    axes[1].set_ylabel('Price ($)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('reports/backtest_results.png', dpi=300, bbox_inches='tight')
    logger.info("📊 Plot saved to reports/backtest_results.png")
    plt.close()


def main():
    """Main backtesting entry point."""
    # Load test data
    logger.info("Loading test data...")
    data = pd.read_csv('data/processed/features.csv')
    
    test_start = int(len(data) * 0.85)
    test_data = data.iloc[test_start:].reset_index(drop=True)
    logger.info(f"Test data: {len(test_data)} rows")
    
    Path("reports").mkdir(exist_ok=True)
    
    # Backtest
    metrics, portfolio_history, trades_history = backtest_agent(
        model_path='models/ppo_trading_final',
        test_data=test_data,
        initial_capital=10000.0
    )
    
    # Save results
    portfolio_history.to_csv('reports/portfolio_history.csv', index=False)
    if not trades_history.empty:
        trades_history.to_csv('reports/trades_history.csv', index=False)
    
    logger.info("\n✅ Backtesting complete!")
    logger.info("Results saved to reports/")


if __name__ == "__main__":
    main()
