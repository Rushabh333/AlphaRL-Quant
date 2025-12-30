"""Train RL trading agent."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from src.models.trading_env import TradingEnvironment
from src.utils.logger import get_logger

logger = get_logger(__name__)


def create_train_test_split(data: pd.DataFrame, train_ratio: float = 0.7, val_ratio: float = 0.15):
    """Split data into train/validation/test sets."""
    n = len(data)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    train_data = data.iloc[:train_end].reset_index(drop=True)
    val_data = data.iloc[train_end:val_end].reset_index(drop=True)
    test_data = data.iloc[val_end:].reset_index(drop=True)
    
    logger.info(f"Data split: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")
    return train_data, val_data, test_data


def train_trading_agent(total_timesteps: int = 10_000):
    """Train RL trading agent with PPO."""
    logger.info("=" * 80)
    logger.info("STARTING RL AGENT TRAINING")
    logger.info("=" * 80)
    
    # Load data
    logger.info("Loading feature data...")
    data = pd.read_csv('data/processed/features.csv')
    logger.info(f"Loaded {len(data)} rows with {len(data.columns)} columns")
    
    # Split data
    train_data, val_data, test_data = create_train_test_split(data)
    
    logger.info("Creating training environment...")
    train_env = TradingEnvironment(
        data=train_data,
        initial_capital=10000.0,
        transaction_cost=0.001,
        reward_type='sharpe',
        action_type='discrete'
    )
    train_env = Monitor(train_env)
    train_env = DummyVecEnv([lambda: train_env])
    
    logger.info("Creating validation environment...")
    eval_env = TradingEnvironment(
        data=val_data,
        initial_capital=10000.0,
        transaction_cost=0.001,
        reward_type='sharpe',
        action_type='discrete'
    )
    eval_env = Monitor(eval_env)
    
    Path("models/best").mkdir(parents=True, exist_ok=True)
    Path("models/checkpoints").mkdir(parents=True, exist_ok=True)
    Path("logs/eval").mkdir(parents=True, exist_ok=True)
    Path("logs/tensorboard").mkdir(parents=True, exist_ok=True)
    
    # Callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path='./models/best/',
        log_path='./logs/eval/',
        eval_freq=5000,
        deterministic=True,
        render=False,
        verbose=1
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=25000,
        save_path='./models/checkpoints/',
        name_prefix='ppo_trading'
    )
    
    callback = CallbackList([eval_callback, checkpoint_callback])
    
    logger.info("Initializing PPO agent...")
    model = PPO(
        'MlpPolicy',
        train_env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=dict(net_arch=[256, 256]),
        verbose=1,
        tensorboard_log="./logs/tensorboard/"
    )
    
    # Train
    logger.info("=" * 80)
    logger.info(f"Training PPO for {total_timesteps:,} timesteps")
    logger.info("=" * 80)
    
    model.learn(
        total_timesteps=total_timesteps,
        callback=callback,
        progress_bar=True
    )
    
    # Save final model
    model_path = "models/ppo_trading_final"
    model.save(model_path)
    logger.info(f"✅ Training complete! Model saved to {model_path}")
    
    return model, test_data


def main():
    """Main training entry point."""
    logger.info("Running quick training test (10K steps)...")
    model, test_data = train_trading_agent(total_timesteps=10_000)
    
    logger.info("\n" + "=" * 80)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 80)
    logger.info("Next steps:")
    logger.info("1. TensorBoard: tensorboard --logdir=./logs/tensorboard/")
    logger.info("2. Backtest: python src/evaluation/backtest.py")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
