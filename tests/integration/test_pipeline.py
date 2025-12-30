"""
Integration tests for the full pipeline.
"""
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from scripts.run_pipeline import MLPipeline
from src.data.collectors import YahooFinanceCollector
from src.data.processors import DataProcessor
from src.features.engineering import FeatureEngineer


def create_mock_data(tickers=['AAPL'], n_days=100):
    """Create mock market data for testing."""
    dates = pd.date_range('2023-01-01', periods=n_days)
    
    all_data = []
    for ticker in tickers:
        data = pd.DataFrame({
            'Date': dates,
            'Open': np.random.uniform(100, 200, n_days),
            'High': np.random.uniform(150, 250, n_days),
            'Low': np.random.uniform(50, 150, n_days),
            'Close': np.random.uniform(100, 200, n_days),
            'Volume': np.random.randint(1000000, 10000000, n_days),
            'ticker': ticker
        })
        # Ensure High >= Low
        data['High'] = data[['High', 'Low', 'Open', 'Close']].max(axis=1)
        data['Low'] = data[['Low', 'Open', 'Close']].min(axis=1)
        all_data.append(data)
    
    return pd.concat(all_data, ignore_index=True)


class TestPipelineIntegration:
    """Integration tests for the full pipeline."""
    
    @patch('src.data.collectors.yf.download')
    def test_full_pipeline_execution(self, mock_download):
        """Test complete pipeline end-to-end."""
        # Mock Yahoo Finance response
        mock_download.return_value = create_mock_data(tickers=['AAPL']).set_index('Date')
        
        # Run pipeline
        pipeline = MLPipeline()
        result = pipeline.run_full_pipeline(save_to_db=False)
        
        # Assertions
        assert result is not None
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
        assert 'SMA_20' in result.columns
        assert 'RSI' in result.columns
        assert 'MACD' in result.columns
    
    @patch('src.data.collectors.yf.download')
    def test_pipeline_with_multiple_tickers(self, mock_download):
        """Test pipeline with multiple tickers."""
        mock_download.return_value = create_mock_data(
            tickers=['AAPL', 'GOOGL']
        ).set_index('Date')
        
        pipeline = MLPipeline()
        result = pipeline.run_full_pipeline(save_to_db=False)
        
        assert result['ticker'].nunique() >= 1  # At least one ticker
        assert 'SMA_20' in result.columns
    
    def test_data_collection_to_processing(self):
        """Test data collection -> processing integration."""
        # Create mock data
        data = create_mock_data()
        
        # Process
        processor = DataProcessor(data)
        processed = processor.process()
        
        assert len(processed) > 0
        assert not processed.empty
    
    def test_processing_to_feature_engineering(self):
        """Test processing -> feature engineering integration."""
        # Create and process data
        data = create_mock_data()
        processor = DataProcessor(data)
        processed = processor.process()
        
        # Engineer features
        engineer = FeatureEngineer(processed)
        features = engineer.engineer_features()
        
        assert len(features) > 0
        assert 'SMA_20' in features.columns
        assert 'return_1d' in features.columns
    
    def test_pipeline_error_handling(self):
        """Test pipeline handles errors gracefully."""
        pipeline = MLPipeline()
        
        # Should raise error when no data
        with pytest.raises(Exception):
            pipeline.run_data_processing()  # No data collected yet


class TestDataValidation:
    """Test data validation across pipeline stages."""
    
    def test_pipeline_handles_missing_data_gracefully(self):
        """Test pipeline behavior when API returns partial data."""
        call_count = [0]
        
        def mock_download_with_failures(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 2:  # Second call fails
                raise ConnectionError("Network error")
            return create_mock_data(tickers=['AAPL']).set_index('Date')
        
        with patch('src.data.collectors.yf.download', side_effect=mock_download_with_failures):
            collector = YahooFinanceCollector(
                tickers=['AAPL', 'GOOGL', 'MSFT'],
                start_date='2023-01-01',
                end_date='2023-01-31'
            )
            
            # Should continue despite one failure
            data = collector.collect()
            assert 'AAPL' in data['ticker'].values
            # GOOGL should be skipped due to error
    
    def test_pipeline_respects_rate_limiting(self):
        """Test that rate limiting is enforced."""
        import time
        
        with patch('src.data.collectors.yf.download') as mock_download:
            mock_download.return_value = create_mock_data(tickers=['AAPL']).set_index('Date')
            
            collector = YahooFinanceCollector(
                tickers=['AAPL', 'GOOGL'],
                start_date='2023-01-01',
                end_date='2023-01-31'
            )
            
            start = time.time()
            collector.collect()
            elapsed = time.time() - start
            
            # With 2 tickers and 2 calls/sec limit,
            # should take at least 0.5 seconds
            assert elapsed >= 0.4  # Small buffer for timing variance
    
    def test_pipeline_retries_on_temporary_failure(self):
        """Test retry logic with temporary failure."""
        call_count = [0]
        
        def mock_download_with_retry(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] < 2:
                raise ConnectionError("Temporary network issue")
            return create_mock_data(tickers=['AAPL']).set_index('Date')
        
        with patch('src.data.collectors.yf.download', side_effect=mock_download_with_retry):
            collector = YahooFinanceCollector(
                tickers=['AAPL'],
                start_date='2023-01-01',
                end_date='2023-01-31'
            )
            
            # Should succeed after retry
            data = collector.collect()
            assert len(data) > 0
            # Verify retry happened (initial attempt + 1 retry)
            assert call_count[0] == 2
    
    def test_schema_validation_catches_bad_data(self):
        """Test that schema validation catches data quality issues."""
        import pandas as pd
        import pandera as pa
        
        bad_data = pd.DataFrame({
            'Date': pd.to_datetime(['2023-01-01']),
            'Open': [-100.0],  # Negative price
            'High': [150.0],
            'Low': [200.0],  # Low > High (invalid)
            'Close': [150.0],
            'Volume': [-1000],  # Negative volume
            'ticker': ['AAPL']
        })
        
        from src.data.schemas import MarketDataSchema
        
        # Should raise validation error
        with pytest.raises(pa.errors.SchemaError):
            MarketDataSchema.validate(bad_data)


class TestDataValidation:
    """Test data validation across pipeline stages."""
    
    def test_invalid_data_rejected(self):
        """Test that invalid data is rejected."""
        invalid_data = pd.DataFrame({'ticker': ['AAPL']})
        
        with pytest.raises(ValueError) as exc_info:
            processor = DataProcessor(invalid_data)
        
        # Check error message is helpful
        assert "missing required columns" in str(exc_info.value).lower()
    
    def test_empty_data_rejected(self):
        """Test that empty data is rejected."""
        empty_data = pd.DataFrame()
        
        with pytest.raises((ValueError, TypeError)) as exc_info:
            processor = DataProcessor(empty_data)
        
        # Check error message provides context
        assert "dataframe" in str(exc_info.value).lower() or "rows" in str(exc_info.value).lower()
    
    def test_helpful_error_messages(self):
        """Test that error messages include helpful hints."""
        # Missing columns
        bad_data = pd.DataFrame({
            'ticker': ['AAPL'],
            'Close': [150.0]
        })
        
        with pytest.raises(ValueError) as exc_info:
            processor = DataProcessor(bad_data)
        
        error_msg = str(exc_info.value)
        # Should include hint about what went wrong
        assert "Hint:" in error_msg or "hint" in error_msg.lower()


class TestHealthChecks:
    """Test health check system."""
    
    def test_health_check_runs(self):
        """Test that health checks can be executed."""
        from src.utils.health import health_checker
        
        status = health_checker.get_status()
        
        assert status is not None
        assert hasattr(status, 'healthy')
        assert hasattr(status, 'checks')
        assert isinstance(status.checks, dict)
    
    def test_health_status_serializable(self):
        """Test that health status can be serialized to JSON."""
        from src.utils.health import health_checker
        import json
        
        status = health_checker.get_status()
        status_dict = status.to_dict()
        
        # Should be JSON serializable
        json_str = json.dumps(status_dict)
        assert json_str is not None
