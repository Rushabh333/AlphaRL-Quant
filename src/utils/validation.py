"""
Input validation utilities for the ML pipeline.
Ensures data quality and prevents silent failures.
"""
import pandas as pd
from typing import List
from src.utils.logger import get_logger
from src.utils.exceptions import DataValidationError

logger = get_logger(__name__)


def validate_dataframe(
    data: pd.DataFrame,
    required_columns: List[str] = None,
    min_rows: int = 1,
    class_name: str = "DataProcessor"
) -> None:
    """Validate DataFrame input for processing classes."""
    # Type check with helpful hint
    if not isinstance(data, pd.DataFrame):
        raise TypeError(
            f"{class_name} requires a pandas DataFrame.\n"
            f"Got: {type(data).__name__}\n"
            f"Hint: Make sure you're passing the data variable, not the class."
        )
    
    # Empty check with context
    if data.empty or len(data) < min_rows:
        raise ValueError(
            f"{class_name} requires at least {min_rows} rows.\n"
            f"Got: {len(data)} rows\n"
            f"Hint: Check if data collection succeeded. "
            f"This might indicate an API issue or date range problem."
        )
    
    # Column check with suggestions
    if required_columns:
        missing_columns = set(required_columns) - set(data.columns)
        if missing_columns:
            available = sorted(data.columns)
            raise ValueError(
                f"{class_name} is missing required columns: {missing_columns}\n"
                f"Available columns: {available}\n"
                f"Hint: This usually means data collection didn't complete properly.\n"
                f"      Check the logs for data collection errors."
            )
    
    logger.debug(f"✓ Validation passed for {class_name}: {len(data)} rows, {len(data.columns)} columns")


def validate_ticker_data(data: pd.DataFrame) -> None:
    """Validate that DataFrame contains proper ticker-based time series data."""
    validate_dataframe(
        data,
        required_columns=['ticker', 'Date'],
        min_rows=1,
        class_name="TickerDataValidator"
    )
    
    # Check Date column is datetime
    if not pd.api.types.is_datetime64_any_dtype(data['Date']):
        raise ValueError("Date column must be datetime type")
    
    # Check tickers exist
    if data['ticker'].nunique() == 0:
        raise ValueError("No tickers found in data")
    
    logger.debug(f"Ticker validation passed: {data['ticker'].nunique()} tickers")
