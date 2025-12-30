"""
AlphaRL-Quant Configuration Management
Centralized configuration for reproducibility and maintainability.
"""
from dataclasses import dataclass, field
from typing import List, Optional
from pathlib import Path
import os


@dataclass
class DataConfig:
    """Data collection and processing configuration."""
    # Data source
    source: str = "yahoo"
    tickers: List[str] = field(default_factory=lambda: ["AAPL", "GOOGL", "MSFT"])
    start_date: str = "2020-01-01"
    end_date: str = "2024-12-31"
    
    # Paths (using pathlib for cross-platform compatibility)
    cache_dir: Path = field(default_factory=lambda: Path("./data/cache"))
    raw_dir: Path = field(default_factory=lambda: Path("./data/raw"))
    processed_dir: Path = field(default_factory=lambda: Path("./data/processed"))
    
    # Processing
    train_ratio: float = 0.7
    val_ratio: float = 0.15  # test = 1 - train - val
    
    def __post_init__(self):
        """Ensure all directories exist."""
        for dir_path in [self.cache_dir, self.raw_dir, self.processed_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)


@dataclass
class FeatureConfig:
    """Feature engineering configuration."""
    # Technical indicators
    sma_periods: List[int] = field(default_factory=lambda: [20, 50, 200])
    ema_periods: List[int] = field(default_factory=lambda: [12, 26])
    rsi_period: int = 14
    
    # MACD
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    
    # Bollinger Bands
    bb_period: int = 20
    bb_std: int = 2
    
    # Lag features
    create_lags: bool = True
    lag_columns: List[str] = field(default_factory=lambda: ["Close", "Volume"])
    lag_periods: List[int] = field(default_factory=lambda: [1, 2, 3])
    
    # Lookback window for RL environment
    lookback_window: int = 60


@dataclass
class RLEnvironmentConfig:
    """RL trading environment configuration."""
    initial_capital: float = 10000.0
    transaction_cost: float = 0.001  # 0.1% per trade
    reward_type: str = "sharpe"  # 'simple', 'sharpe', 'sortino'
    window_size: int = 30  # Lookback for state representation
    action_type: str = "discrete"  # 'discrete' or 'continuous'
    
    # Risk management (future enhancement)
    max_position_size: float = 1.0  # 100% of capital
    stop_loss_pct: Optional[float] = None  # e.g., 0.05 for 5% stop loss


@dataclass
class TrainingConfig:
    """RL agent training configuration."""
    # Algorithm
    algorithm: str = "PPO"  # 'PPO', 'A2C', 'SAC'
    
    # Training hyperparameters
    total_timesteps: int = 1_000_000
    learning_rate: float = 3e-4
    n_steps: int = 2048  # Steps per update
    batch_size: int = 64
    n_epochs: int = 10
    
    # PPO-specific
    gamma: float = 0.99  # Discount factor
    gae_lambda: float = 0.95  # GAE parameter
    clip_range: float = 0.2  # PPO clipping
    ent_coef: float = 0.01  # Entropy coefficient (exploration)
    vf_coef: float = 0.5  # Value function coefficient
    max_grad_norm: float = 0.5  # Gradient clipping
    
    # Network architecture
    policy_net_arch: List[int] = field(default_factory=lambda: [256, 256])
    
    # Callbacks & monitoring
    eval_freq: int = 5000  # Evaluate every N steps
    checkpoint_freq: int = 25000  # Save every N steps
    
    # Paths
    model_dir: Path = field(default_factory=lambda: Path("./models"))
    log_dir: Path = field(default_factory=lambda: Path("./logs"))
    tensorboard_log: Path = field(default_factory=lambda: Path("./logs/tensorboard"))
    
    def __post_init__(self):
        """Ensure all directories exist."""
        for dir_path in [self.model_dir, self.log_dir, self.tensorboard_log]:
            dir_path.mkdir(parents=True, exist_ok=True)


@dataclass
class PipelineConfig:
    """Overall pipeline configuration."""
    # Error handling
    fail_on_db_error: bool = False
    fail_on_validation_error: bool = True
    max_retries: int = 3
    
    # API rate limiting
    rate_limit_per_second: float = 2.0
    
    # Health checks
    enable_health_checks: bool = True
    
    # Reproducibility
    random_seed: int = 42


@dataclass
class DatabaseConfig:
    """Database configuration."""
    host: str = "localhost"
    port: int = 5432
    name: str = "trading_db"
    user: str = "postgres"
    password: str = field(default_factory=lambda: os.getenv("DB_PASSWORD", ""))
    pool_size_min: int = 1
    pool_size_max: int = 10


@dataclass
class AlphaRLConfig:
    """Master configuration class combining all sub-configurations."""
    data: DataConfig = field(default_factory=DataConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    rl_env: RLEnvironmentConfig = field(default_factory=RLEnvironmentConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    
    @classmethod
    def from_yaml(cls, config_path: str) -> 'AlphaRLConfig':
        """
        Load configuration from YAML file.
        Falls back to defaults if file doesn't exist.
        """
        import yaml
        
        try:
            with open(config_path, 'r') as f:
                config_dict = yaml.safe_load(f)
            
            return cls(
                data=DataConfig(**config_dict.get('data', {})),
                features=FeatureConfig(**config_dict.get('features', {})),
                rl_env=RLEnvironmentConfig(**config_dict.get('rl_env', {})),
                training=TrainingConfig(**config_dict.get('training', {})),
                pipeline=PipelineConfig(**config_dict.get('pipeline', {})),
                database=DatabaseConfig(**config_dict.get('database', {}))
            )
        except FileNotFoundError:
            print(f"Config file {config_path} not found. Using defaults.")
            return cls()
    
    def save_yaml(self, config_path: str):
        """Save current configuration to YAML file."""
        import yaml
        from dataclasses import asdict
        
        config_dict = asdict(self)
        
        # Convert Path objects to strings for YAML serialization
        def path_to_str(obj):
            if isinstance(obj, Path):
                return str(obj)
            elif isinstance(obj, dict):
                return {k: path_to_str(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [path_to_str(item) for item in obj]
            return obj
        
        config_dict = path_to_str(config_dict)
        
        with open(config_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)


# Global config instance (can be overridden)
config = AlphaRLConfig()


def seed_everything(seed: int = 42):
    """
    Set random seeds for reproducibility across all libraries.
    
    Critical for:
    - Reproducible train/test splits
    - Consistent RL agent behavior
    - Debugging (ability to replay exact scenarios)
    
    Args:
        seed: Random seed value
    """
    import random
    import numpy as np
    import torch
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Make CUDA operations deterministic (slight performance cost)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    print(f"Random seed set to {seed} for reproducibility")
