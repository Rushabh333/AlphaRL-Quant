"""
Data processing module for cleaning and preprocessing.
"""
import pandas as pd
import numpy as np
from typing import Optional, Dict
from src.utils.logger import get_logger, log_execution_time
from src.utils.exceptions import DataValidationError
from src.utils.validation import validate_dataframe

logger = get_logger(__name__)


class DataProcessor:
    """
    Handles data cleaning and preprocessing.
    Modular design allows easy addition of new processing steps.
    """
    
    REQUIRED_COLUMNS = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'ticker']
    
    def __init__(self, data: pd.DataFrame):
        # Input validation
        validate_dataframe(
            data,
            required_columns=self.REQUIRED_COLUMNS,
            min_rows=10,
            class_name="DataProcessor"
        )
        
        self.data = data.copy()
        self.original_shape = data.shape
        logger.info(f"Initialized DataProcessor with {self.original_shape[0]} rows")
    
    @log_execution_time
    def handle_missing_values(self, method: str = 'ffill') -> 'DataProcessor':
    """Handle missing values."""
        logger.info(f"Handling missing values with method: {method}")
        
        missing_before = self.data.isnull().sum().sum()
        
        if method == 'ffill':
            self.data.fillna(method='ffill', inplace=True)
        elif method == 'bfill':
            self.data.fillna(method='bfill', inplace=True)
        elif method == 'interpolate':
            self.data.interpolate(method='linear', inplace=True)
        elif method == 'drop':
            self.data.dropna(inplace=True)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        missing_after = self.data.isnull().sum().sum()
        logger.info(f"Reduced missing values from {missing_before} to {missing_after}")
        
        return self
    
    @log_execution_time
    def remove_outliers(self, columns: list, method: str = 'iqr', threshold: float = 3.0) -> 'DataProcessor':
    """Remove outliers from specified columns."""
        logger.info(f"Removing outliers using {method} method")
        
        rows_before = len(self.data)
        
        for col in columns:
            if col not in self.data.columns:
                logger.warning(f"Column {col} not found, skipping")
                continue
            
            if method == 'iqr':
                Q1 = self.data[col].quantile(0.25)
                Q3 = self.data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                
                mask = (self.data[col] >= lower_bound) & (self.data[col] <= upper_bound)
                
            elif method == 'zscore':
                z_scores = np.abs((self.data[col] - self.data[col].mean()) / self.data[col].std())
                mask = z_scores < threshold
            
            else:
                raise ValueError(f"Unknown method: {method}")
            
            outliers_removed = (~mask).sum()
            self.data = self.data[mask]
            logger.debug(f"Removed {outliers_removed} outliers from {col}")
        
        rows_after = len(self.data)
        logger.info(f"Removed {rows_before - rows_after} rows ({(rows_before - rows_after) / rows_before * 100:.2f}%)")
        
        return self
    
    @log_execution_time
    def normalize_columns(self, columns: list, method: str = 'minmax') -> 'DataProcessor':
    """Normalize specified columns."""
        logger.info(f"Normalizing columns using {method} method")
        
        for col in columns:
            if col not in self.data.columns:
                logger.warning(f"Column {col} not found, skipping")
                continue
            
            if method == 'minmax':
                min_val = self.data[col].min()
                max_val = self.data[col].max()
                self.data[col] = (self.data[col] - min_val) / (max_val - min_val)
                
            elif method == 'zscore':
                mean = self.data[col].mean()
                std = self.data[col].std()
                self.data[col] = (self.data[col] - mean) / std
            
            else:
                raise ValueError(f"Unknown method: {method}")
            
            logger.debug(f"Normalized {col}")
        
        return self
    
    @log_execution_time
    def sort_by_date(self) -> 'DataProcessor':
        """Sort data by date and ticker."""
        if 'Date' not in self.data.columns:
            logger.warning("Date column not found, skipping sort")
            return self
        
        self.data.sort_values(['ticker', 'Date'], inplace=True)
        self.data.reset_index(drop=True, inplace=True)
        logger.info("Sorted data by ticker and date")
        
        return self
    
    def get_processed_data(self) -> pd.DataFrame:
        """Return processed data."""
        logger.info(f"Returning processed data: {self.data.shape[0]} rows (from {self.original_shape[0]})")
        return self.data
    
    @log_execution_time
    def process(self, config: Optional[Dict] = None) -> pd.DataFrame:
        """
        Execute full processing pipeline.
        
        Stages:
        1. Handle missing values
        2. Remove outliers
        3. Sort by date
        4. Return clean data
        """
        logger.info("Starting full data processing pipeline")
        
        # Default config
        if config is None:
            config = {
                'missing_method': 'ffill',
                'outlier_columns': ['Volume'],
                'outlier_method': 'iqr',
                'outlier_threshold': 3.0
            }
        
        try:
            # Chain processing steps
            self.handle_missing_values(method=config.get('missing_method', 'ffill'))
            
            if config.get('outlier_columns'):
                self.remove_outliers(
                    columns=config['outlier_columns'],
                    method=config.get('outlier_method', 'iqr'),
                    threshold=config.get('outlier_threshold', 3.0)
                )
            
            self.sort_by_date()
            
            logger.info("Data processing pipeline completed successfully")
            return self.get_processed_data()
            
        except Exception as e:
            logger.error(f"Data processing failed: {str(e)}")
            raise DataValidationError(f"Processing failed: {str(e)}") from e
