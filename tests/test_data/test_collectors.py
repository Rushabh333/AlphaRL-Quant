"""
Test suite for data collectors.
"""
import pytest
import pandas as pd
from src.data.collectors import YahooFinanceCollector
from src.utils.exceptions import DataCollectionError


class TestYahooFinanceCollector:
    """Test suite for Yahoo Finance data collection."""
    
    @pytest.fixture
    def collector(self):
        """Create a collector instance for testing."""
        return YahooFinanceCollector(
            tickers=['AAPL'],
            start_date='2023-01-01',
            end_date='2023-01-31'
        )
    
    def test_fetch_data_success(self, collector):
        """Test successful data fetching."""
        data = collector.fetch_data()
        
        assert isinstance(data, pd.DataFrame)
        assert len(data) > 0
        assert 'ticker' in data.columns
        assert 'Close' in data.columns
    
    def test_validate_data_valid(self, collector):
        """Test data validation with valid data."""
        data = collector.fetch_data()
        assert collector.validate_data(data) == True
    
    def test_validate_data_missing_columns(self, collector):
        """Test validation fails with missing columns."""
        invalid_data = pd.DataFrame({'Date': [], 'Close': []})
        assert collector.validate_data(invalid_data) == False
    
    def test_collect_empty_tickers(self):
        """Test collector with empty ticker list."""
        collector = YahooFinanceCollector(
            tickers=[],
            start_date='2023-01-01',
            end_date='2023-01-31'
        )
        
        with pytest.raises(DataCollectionError):
            collector.collect()
