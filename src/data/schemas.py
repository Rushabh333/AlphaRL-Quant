"""
Data schema validation using Pandera.
Ensures data quality and catches data drift early.
"""
import pandas as pd
import pandera as pa
from pandera import DataFrameSchema, Column, Check
from src.utils.logger import get_logger

logger = get_logger(__name__)


class MarketDataSchema:
    """
    Schema validator for market data.
    Enforces data types and constraints to prevent silent failures.
    """
    
    schema = DataFrameSchema(
        columns={
            'Date': Column(
                pa.DateTime,
                nullable=False,
                description="Trading date"
            ),
            'Open': Column(
                float,
                checks=[
                    Check.greater_than(0, error="Open price must be positive"),
                    Check.less_than(1000000, error="Open price unreasonably high")
                ],
                nullable=False,
                description="Opening price"
            ),
            'High': Column(
                float,
                checks=[
                    Check.greater_than(0, error="High price must be positive"),
                    Check.less_than(1000000, error="High price unreasonably high")
                ],
                nullable=False,
                description="Highest price"
            ),
            'Low': Column(
                float,
                checks=[
                    Check.greater_than(0, error="Low price must be positive"),
                    Check.less_than(1000000, error="Low price unreasonably high")
                ],
                nullable=False,
                description="Lowest price"
            ),
            'Close': Column(
                float,
                checks=[
                    Check.greater_than(0, error="Close price must be positive"),
                    Check.less_than(1000000, error="Close price unreasonably high")
                ],
                nullable=False,
                description="Closing price"
            ),
            'Volume': Column(
                int,
                checks=[
                    Check.greater_than_or_equal_to(0, error="Volume cannot be negative")
                ],
                nullable=False,
                description="Trading volume"
            ),
            'ticker': Column(
                str,
                nullable=False,
                description="Stock ticker symbol"
            ),
        },
        checks=[
            # Cross-column validation: High >= Low
            Check(
                lambda df: (df['High'] >= df['Low']).all(),
                error="High price must be >= Low price"
            ),
            # Cross-column validation: High >= Open, Close
            Check(
                lambda df: (df['High'] >= df['Open']).all(),
                error="High price must be >= Open price"
            ),
            Check(
                lambda df: (df['High'] >= df['Close']).all(),
                error="High price must be >= Close price"
            ),
            # Cross-column validation: Low <= Open, Close
            Check(
                lambda df: (df['Low'] <= df['Open']).all(),
                error="Low price must be <= Open price"
            ),
            Check(
                lambda df: (df['Low'] <= df['Close']).all(),
                error="Low price must be <= Close price"
            ),
        ],
        strict=True,  # Don't allow extra columns
        coerce=True   # Try to coerce types when possible
    )
    
    @classmethod
    def validate(cls, data: pd.DataFrame) -> pd.DataFrame:
    """Validate market data against schema."""
        logger.info("Validating market data schema")
        try:
            validated_data = cls.schema.validate(data, lazy=False)
            logger.info(f"✓ Schema validation passed for {len(validated_data)} rows")
            return validated_data
        except pa.errors.SchemaError as e:
            logger.error(f"✗ Schema validation failed: {e}")
            raise
    
    @classmethod
    def validate_lazy(cls, data: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    """Validate with lazy evaluation to collect all errors."""
        logger.info("Validating market data schema (lazy mode)")
        try:
            validated_data = cls.schema.validate(data, lazy=True)
            logger.info(f"✓ Schema validation passed for {len(validated_data)} rows")
            return validated_data, []
        except pa.errors.SchemaErrors as e:
            logger.error(f"✗ Schema validation failed with {len(e.failure_cases)} errors")
            return data, e.failure_cases


class FeatureDataSchema:
    """
    Schema validator for engineered features.
    Validates that features are properly created and within expected ranges.
    """
    
    @staticmethod
    def validate_features(data: pd.DataFrame, feature_names: list) -> pd.DataFrame:
    """Validate that all expected features exist and contain valid values."""
        logger.info(f"Validating {len(feature_names)} features")
        
        # Check for missing features
        missing_features = set(feature_names) - set(data.columns)
        if missing_features:
            raise ValueError(f"Missing features: {missing_features}")
        
        # Check for inf values
        inf_cols = []
        for col in feature_names:
            if data[col].isin([float('inf'), float('-inf')]).any():
                inf_cols.append(col)
        
        if inf_cols:
            logger.warning(f"Features with inf values: {inf_cols}")
            # Replace inf with NaN
            data[inf_cols] = data[inf_cols].replace([float('inf'), float('-inf')], pd.NA)
        
        # Check percentage of NaN values
        nan_pct = data[feature_names].isnull().sum() / len(data)
        high_nan_features = nan_pct[nan_pct > 0.5].index.tolist()
        
        if high_nan_features:
            logger.warning(f"Features with >50% NaN: {high_nan_features}")
        
        logger.info("✓ Feature validation completed")
        return data
