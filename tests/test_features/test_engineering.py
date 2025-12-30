"""
Test suite for feature engineering.
"""
import pytest
import pandas as pd
import numpy as np
from src.features.engineering import FeatureEngineer


class TestFeatureEngineer:
    """Test suite for feature engineering."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample market data for testing."""
        dates = pd.date_range('2023-01-01', periods=100)
        data = pd.DataFrame({
            'Date': dates,
            'ticker': 'AAPL',
            'Open': np.random.uniform(100, 200, 100),
            'High': np.random.uniform(100, 200, 100),
            'Low': np.random.uniform(100, 200, 100),
            'Close': np.random.uniform(100, 200, 100),
            'Volume': np.random.randint(1000000, 10000000, 100)
        })
        return data
    
    def test_add_technical_indicators(self, sample_data):
        """Test adding technical indicators."""
        engineer = FeatureEngineer(sample_data)
        engineer.add_technical_indicators()
        
        data = engineer.get_feature_data()
        
        # Check if SMA columns exist
        assert 'SMA_20' in data.columns
        assert 'RSI' in data.columns
        assert 'MACD' in data.columns
    
    def test_add_price_features(self, sample_data):
        """Test adding price features."""
        engineer = FeatureEngineer(sample_data)
        engineer.add_price_features()
        
        data = engineer.get_feature_data()
        
        assert 'return_1d' in data.columns
        assert 'log_return' in data.columns
        assert 'volatility_10d' in data.columns
    
    def test_feature_names_tracked(self, sample_data):
        """Test that feature names are properly tracked."""
        engineer = FeatureEngineer(sample_data)
        engineer.add_technical_indicators()
        
        assert len(engineer.feature_names) > 0
        assert 'SMA_20' in engineer.feature_names
