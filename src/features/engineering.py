"""
Feature engineering module for trading data.
Optimized with groupby operations for better performance.
"""
import pandas as pd
import numpy as np
import ta  # Technical Analysis library
from typing import List, Dict, Optional
from src.utils.logger import get_logger, log_execution_time
from src.utils.exceptions import FeatureEngineeringError
from src.utils.validation import validate_ticker_data

logger = get_logger(__name__)


class FeatureEngineer:
    """
    Modular feature engineering for trading data.
    Uses groupby.transform for efficient per-ticker calculations.
    """
    
    def __init__(self, data: pd.DataFrame):
        # Input validation
        validate_ticker_data(data)
        
        self.data = data.copy()
        self.feature_names = []
        logger.info(f"Initialized FeatureEngineer with {len(data)} rows")
    
    @log_execution_time
    def add_technical_indicators(self, config: Optional[Dict] = None) -> 'FeatureEngineer':
        """
        Add technical indicators.
        
        Calculates:
        - SMA (Simple Moving Average)
        - EMA (Exponential Moving Average)
        - RSI (Relative Strength Index)
        - MACD (Moving Average Convergence Divergence)
        - Bollinger Bands
        """
        logger.info("Adding technical indicators")
        
        if config is None:
            config = {
                'sma_periods': [20, 50, 200],
                'ema_periods': [12, 26],
                'rsi_period': 14,
                'macd_fast': 12,
                'macd_slow': 26,
                'macd_signal': 9,
                'bb_period': 20,
                'bb_std': 2
            }
        
        try:
            # Group by ticker for efficient per-stock calculations
            grouped = self.data.groupby('ticker', group_keys=False)
            
            # Simple Moving Averages - using transform for performance
            for period in config['sma_periods']:
                col_name = f'SMA_{period}'
                self.data[col_name] = grouped['Close'].transform(
                    lambda x: x.rolling(window=period).mean()
                )
                if col_name not in self.feature_names:
                    self.feature_names.append(col_name)
            
            # Exponential Moving Averages
            for period in config['ema_periods']:
                col_name = f'EMA_{period}'
                self.data[col_name] = grouped['Close'].transform(
                    lambda x: x.ewm(span=period, adjust=False).mean()
                )
                if col_name not in self.feature_names:
                    self.feature_names.append(col_name)
            
            # For technical indicators that don't support transform directly,
            # use apply with concatenation (still faster than manual looping)
            def add_rsi(group):
                group['RSI'] = ta.momentum.RSIIndicator(
                    group['Close'],
                    window=config['rsi_period']
                ).rsi()
                return group
            
            def add_macd(group):
                macd = ta.trend.MACD(
                    group['Close'],
                    window_fast=config['macd_fast'],
                    window_slow=config['macd_slow'],
                    window_sign=config['macd_signal']
                )
                group['MACD'] = macd.macd()
                group['MACD_signal'] = macd.macd_signal()
                group['MACD_diff'] = macd.macd_diff()
                return group
            
            def add_bollinger_bands(group):
                bb = ta.volatility.BollingerBands(
                    group['Close'],
                    window=config['bb_period'],
                    window_dev=config['bb_std']
                )
                group['BB_high'] = bb.bollinger_hband()
                group['BB_low'] = bb.bollinger_lband()
                group['BB_mid'] = bb.bollinger_mavg()
                group['BB_width'] = bb.bollinger_wband()
                return group
            
            # Apply indicators
            self.data = grouped.apply(add_rsi)
            for col in ['RSI']:
                if col not in self.feature_names:
                    self.feature_names.append(col)
            
            self.data = self.data.groupby('ticker', group_keys=False).apply(add_macd)
            for col in ['MACD', 'MACD_signal', 'MACD_diff']:
                if col not in self.feature_names:
                    self.feature_names.append(col)
            
            self.data = self.data.groupby('ticker', group_keys=False).apply(add_bollinger_bands)
            for col in ['BB_high', 'BB_low', 'BB_mid', 'BB_width']:
                if col not in self.feature_names:
                    self.feature_names.append(col)
            
            logger.info(f"Added {len(self.feature_names)} technical indicators")
            return self
            
        except Exception as e:
            logger.error(f"Failed to add technical indicators: {str(e)}")
            raise FeatureEngineeringError(f"Technical indicators failed: {str(e)}") from e
    
    @log_execution_time
    def add_price_features(self) -> 'FeatureEngineer':
        """
        Add price-based features.
        
        Features:
        - Returns (1-day, 5-day, 20-day)
        - Log returns
        - Price momentum
        - Volatility
        """
        logger.info("Adding price features")
        
        try:
            grouped = self.data.groupby('ticker', group_keys=False)
            
            # Returns - optimized with transform
            for period in [1, 5, 20]:
                col_name = f'return_{period}d'
                self.data[col_name] = grouped['Close'].transform(
                    lambda x: x.pct_change(periods=period)
                )
                if col_name not in self.feature_names:
                    self.feature_names.append(col_name)
            
            # Log returns
            self.data['log_return'] = grouped['Close'].transform(
                lambda x: np.log(x / x.shift(1))
            )
            if 'log_return' not in self.feature_names:
                self.feature_names.append('log_return')
            
            # Volatility (rolling std of returns)
            # First calculate return_1d if not already done
            if 'return_1d' not in self.data.columns:
                self.data['return_1d'] = grouped['Close'].transform(
                    lambda x: x.pct_change()
                )
            
            for period in [10, 20, 30]:
                col_name = f'volatility_{period}d'
                self.data[col_name] = grouped['return_1d'].transform(
                    lambda x: x.rolling(window=period).std()
                )
                if col_name not in self.feature_names:
                    self.feature_names.append(col_name)
            
            # Price momentum
            self.data['momentum'] = grouped['Close'].transform(
                lambda x: x - x.shift(10)
            )
            if 'momentum' not in self.feature_names:
                self.feature_names.append('momentum')
            
            # High-Low range
            self.data['hl_range'] = (self.data['High'] - self.data['Low']) / self.data['Close']
            if 'hl_range' not in self.feature_names:
                self.feature_names.append('hl_range')
            
            logger.info(f"Added price features")
            return self
            
        except Exception as e:
            logger.error(f"Failed to add price features: {str(e)}")
            raise FeatureEngineeringError(f"Price features failed: {str(e)}") from e
    
    @log_execution_time
    def add_volume_features(self) -> 'FeatureEngineer':
        """Add volume-based features."""
        logger.info("Adding volume features")
        
        try:
            grouped = self.data.groupby('ticker', group_keys=False)
            
            # Volume moving averages - optimized
            for period in [10, 20]:
                col_name = f'volume_ma_{period}'
                self.data[col_name] = grouped['Volume'].transform(
                    lambda x: x.rolling(window=period).mean()
                )
                if col_name not in self.feature_names:
                    self.feature_names.append(col_name)
            
            # Volume ratio
            if 'volume_ma_20' in self.data.columns:
                self.data['volume_ratio'] = self.data['Volume'] / self.data['volume_ma_20']
                if 'volume_ratio' not in self.feature_names:
                    self.feature_names.append('volume_ratio')
            
            # On-Balance Volume (OBV) - needs apply
            def add_obv(group):
                group['OBV'] = ta.volume.OnBalanceVolumeIndicator(
                    group['Close'],
                    group['Volume']
                ).on_balance_volume()
                return group
            
            self.data = grouped.apply(add_obv)
            if 'OBV' not in self.feature_names:
                self.feature_names.append('OBV')
            
            logger.info("Added volume features")
            return self
            
        except Exception as e:
            logger.error(f"Failed to add volume features: {str(e)}")
            raise FeatureEngineeringError(f"Volume features failed: {str(e)}") from e
    
    @log_execution_time
    def create_lagged_features(self, columns: List[str], lags: List[int]) -> 'FeatureEngineer':
    """Create lagged features for time series."""
        logger.info(f"Creating lagged features for {len(columns)} columns with lags {lags}")
        
        try:
            grouped = self.data.groupby('ticker', group_keys=False)
            
            for col in columns:
                if col not in self.data.columns:
                    logger.warning(f"Column {col} not found, skipping")
                    continue
                
                for lag in lags:
                    col_name = f'{col}_lag_{lag}'
                    self.data[col_name] = grouped[col].transform(lambda x: x.shift(lag))
                    if col_name not in self.feature_names:
                        self.feature_names.append(col_name)
            
            logger.info(f"Created {len(columns) * len(lags)} lagged features")
            return self
            
        except Exception as e:
            logger.error(f"Failed to create lagged features: {str(e)}")
            raise FeatureEngineeringError(f"Lagged features failed: {str(e)}") from e
    
    def get_feature_data(self) -> pd.DataFrame:
        """Return data with engineered features."""
        logger.info(f"Returning feature data with {len(self.feature_names)} features")
        return self.data
    
    @log_execution_time
    def engineer_features(self, config: Optional[Dict] = None) -> pd.DataFrame:
        """
        Execute full feature engineering pipeline.
        
        Steps:
        1. Add technical indicators
        2. Add price features
        3. Add volume features
        4. Create lagged features (optional)
        5. Handle any remaining NaN from feature creation
        6. Return feature-rich dataset
        """
        logger.info("Starting full feature engineering pipeline")
        
        try:
            # Add all feature categories
            self.add_technical_indicators()
            self.add_price_features()
            self.add_volume_features()
            
            # Optional: Create lagged features
            if config and config.get('create_lags'):
                self.create_lagged_features(
                    columns=config.get('lag_columns', ['Close', 'Volume']),
                    lags=config.get('lag_periods', [1, 2, 3])
                )
            
            # Drop rows with NaN (from indicators that need warmup period)
            rows_before = len(self.data)
            self.data.dropna(inplace=True)
            rows_after = len(self.data)
            logger.info(f"Dropped {rows_before - rows_after} rows due to NaN from feature creation")
            
            logger.info(f"Feature engineering completed. Total features: {len(self.feature_names)}")
            return self.get_feature_data()
            
        except Exception as e:
            logger.error(f"Feature engineering pipeline failed: {str(e)}")
            raise FeatureEngineeringError(f"Pipeline failed: {str(e)}") from e
