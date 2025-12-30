"""
Trading environment for reinforcement learning.
Implements a realistic trading simulator with transaction costs and portfolio tracking.
"""
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from src.utils.logger import get_logger

logger = get_logger(__name__)


class TradingEnvironment(gym.Env):
    """
    Custom RL environment for algorithmic trading.
    
    Implements a realistic trading simulator with:
    - Transaction costs
    - Continuous or discrete action spaces
    - Multiple reward formulations
    - Portfolio tracking
    """
    
    metadata = {'render.modes': ['human']}
    
    def __init__(
        self,
        data: pd.DataFrame,
        initial_capital: float = 10000.0,
        transaction_cost: float = 0.001,  # 0.1%
        reward_type: str = 'sharpe',  # 'simple', 'sharpe', 'sortino'
        window_size: int = 30,  # Lookback window for state
        action_type: str = 'discrete'  # 'discrete' or 'continuous'
    ):
        super().__init__()
        
        # Validate data
        required_cols = ['Close', 'Open', 'High', 'Low', 'Volume']
        missing = set(required_cols) - set(data.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        self.data = data.reset_index(drop=True)
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.reward_type = reward_type
        self.window_size = window_size
        self.action_type = action_type
        
        self.feature_cols = [col for col in data.columns 
                            if col not in ['Date', 'ticker']]
        self.n_features = len(self.feature_cols)
        
        if action_type == 'discrete':
            # 0: Sell, 1: Hold, 2: Buy
            self.action_space = spaces.Discrete(3)
        else:
            # Continuous: -1 (sell all) to 1 (buy all)
            self.action_space = spaces.Box(
                low=-1.0, high=1.0, shape=(1,), dtype=np.float32
            )
        
        # Features + portfolio state (cash, shares, position)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.n_features + 3,),
            dtype=np.float32
        )
        
        # State tracking
        self.current_step = 0
        self.cash = initial_capital
        self.shares = 0
        self.portfolio_value = initial_capital
        self.trades = []
        self.portfolio_history = []
        
        logger.info(f"Trading environment initialized: {len(data)} steps, "
                   f"{self.n_features} features, {action_type} actions")
    
    def reset(self, seed=None, options=None):
        """Reset environment to initial state."""
        if seed is not None:
            np.random.seed(seed)
        
        self.current_step = self.window_size
        self.cash = self.initial_capital
        self.shares = 0
        self.portfolio_value = self.initial_capital
        self.trades = []
        self.portfolio_history = [self.initial_capital]
        
        return self._get_observation(), {}
    
    def _get_observation(self) -> np.ndarray:
        """Get current state observation."""
        row = self.data.iloc[self.current_step]
        features = row[self.feature_cols].values.astype(np.float32)
        
        # Replace any NaN or inf with 0
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Portfolio state (normalized)
        current_price = row['Close']
        position_value = self.shares * current_price
        total_value = self.cash + position_value
        
        portfolio_state = np.array([
            self.cash / self.initial_capital,  # Normalized cash
            position_value / self.initial_capital,  # Normalized position
            self._get_position_type()  # Position type (-1, 0, 1)
        ], dtype=np.float32)
        
        # Combine features and portfolio state
        observation = np.concatenate([features, portfolio_state])
        
        return observation
    
    def _get_position_type(self) -> float:
        """Get position type: -1 (short), 0 (neutral), 1 (long)."""
        if self.shares > 0:
            return 1.0
        elif self.shares < 0:
            return -1.0
        else:
            return 0.0
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one trading step."""
        current_price = self.data.iloc[self.current_step]['Close']
        prev_portfolio_value = self.portfolio_value
        
        # Execute action
        if self.action_type == 'discrete':
            if action == 0:  # Sell
                self._execute_sell(current_price)
            elif action == 2:  # Buy
                self._execute_buy(current_price)
            # action == 1 is Hold
        else:
            # Continuous action: action in [-1, 1]
            self._execute_continuous_action(action[0], current_price)
        
        # Update portfolio value
        self.portfolio_value = self.cash + (self.shares * current_price)
        self.portfolio_history.append(self.portfolio_value)
        
        # Move to next step
        self.current_step += 1
        
        # Calculate reward
        reward = self._calculate_reward(prev_portfolio_value)
        
        done = self.current_step >= len(self.data) - 1
        
        obs = self._get_observation() if not done else self._get_observation()
        
        # Info dict
        info = {
            'portfolio_value': self.portfolio_value,
            'cash': self.cash,
            'shares': self.shares,
            'total_return': (self.portfolio_value / self.initial_capital - 1) * 100
        }
        
        return obs, reward, done, False, info  # terminated, truncated
    
    def _execute_buy(self, price: float):
        """Execute buy order (use 95% of available cash)."""
        available_cash = self.cash * 0.95
        shares_to_buy = int(available_cash / price)
        
        if shares_to_buy > 0:
            cost = shares_to_buy * price * (1 + self.transaction_cost)
            
            if cost <= self.cash:
                self.cash -= cost
                self.shares += shares_to_buy
                
                self.trades.append({
                    'step': self.current_step,
                    'action': 'BUY',
                    'shares': shares_to_buy,
                    'price': price,
                    'cost': cost
                })
    
    def _execute_sell(self, price: float):
        """Execute sell order (sell all shares)."""
        if self.shares > 0:
            revenue = self.shares * price * (1 - self.transaction_cost)
            
            self.trades.append({
                'step': self.current_step,
                'action': 'SELL',
                'shares': self.shares,
                'price': price,
                'revenue': revenue
            })
            
            self.cash += revenue
            self.shares = 0
    
    def _execute_continuous_action(self, action: float, price: float):
        """Execute continuous action."""
        if action < -0.33:  # Sell
            self._execute_sell(price)
        elif action > 0.33:  # Buy
            self._execute_buy(price)
        # else: Hold
    
    def _calculate_reward(self, prev_portfolio_value: float) -> float:
        """Calculate reward based on specified reward type."""
        if self.reward_type == 'simple':
            return (self.portfolio_value / prev_portfolio_value) - 1
        
        elif self.reward_type == 'sharpe':
            if len(self.portfolio_history) < 10:
                return 0.0
            
            recent_values = self.portfolio_history[-10:]
            returns = pd.Series(recent_values).pct_change().dropna()
            
            if len(returns) > 0 and returns.std() > 0:
                sharpe = returns.mean() / returns.std()
                return sharpe
            return 0.0
        
        elif self.reward_type == 'sortino':
            if len(self.portfolio_history) < 10:
                return 0.0
            
            recent_values = self.portfolio_history[-10:]
            returns = pd.Series(recent_values).pct_change().dropna()
            
            if len(returns) > 0:
                downside_returns = returns[returns < 0]
                if len(downside_returns) > 0:
                    downside_std = downside_returns.std()
                    if downside_std > 0:
                        sortino = returns.mean() / downside_std
                        return sortino
            return 0.0
        
        else:
            raise ValueError(f"Unknown reward type: {self.reward_type}")
    
    def render(self, mode='human'):
        """Render current environment state."""
        current_price = self.data.iloc[self.current_step]['Close']
        total_return = (self.portfolio_value / self.initial_capital - 1) * 100
        
        print(f"\n{'='*60}")
        print(f"Step: {self.current_step} / {len(self.data)}")
        print(f"Current Price: ${current_price:.2f}")
        print(f"Cash: ${self.cash:.2f}")
        print(f"Shares: {self.shares}")
        print(f"Portfolio Value: ${self.portfolio_value:.2f}")
        print(f"Total Return: {total_return:.2f}%")
        print(f"Trades Executed: {len(self.trades)}")
        print(f"{'='*60}\n")
    
    def get_portfolio_history(self) -> pd.DataFrame:
        """Get complete portfolio history as DataFrame."""
        return pd.DataFrame({
            'step': range(len(self.portfolio_history)),
            'portfolio_value': self.portfolio_history
        })
    
    def get_trades_history(self) -> pd.DataFrame:
        """Get trade history as DataFrame."""
        if not self.trades:
            return pd.DataFrame()
        return pd.DataFrame(self.trades)
