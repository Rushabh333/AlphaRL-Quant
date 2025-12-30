"""
Custom exceptions for the ML pipeline.
"""


class PipelineException(Exception):
    """Base exception for pipeline errors."""
    pass


class DataCollectionError(PipelineException):
    """Raised when data collection fails."""
    pass


class DataValidationError(PipelineException):
    """Raised when data validation fails."""
    pass


class FeatureEngineeringError(PipelineException):
    """Raised when feature engineering fails."""
    pass


class ModelTrainingError(PipelineException):
    """Raised when model training fails."""
    pass


class BacktestingError(PipelineException):
    """Raised when backtesting fails."""
    pass
