"""
Data validation module.
"""
import pandas as pd
from typing import List, Dict
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DataValidator:
    """
    Validates market data for quality and consistency.
    """
    
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.validation_results = {}
    
    def validate_columns(self, required_columns: List[str]) -> bool:
        """Check if required columns exist."""
        missing_columns = [col for col in required_columns if col not in self.data.columns]
        
        if missing_columns:
            logger.error(f"Missing columns: {missing_columns}")
            self.validation_results['missing_columns'] = missing_columns
            return False
        
        return True
    
    def validate_nulls(self, max_null_pct: float = 0.05) -> bool:
        """Check for excessive null values."""
        null_pct = self.data.isnull().mean()
        excessive_nulls = null_pct[null_pct > max_null_pct]
        
        if not excessive_nulls.empty:
            logger.warning(f"Columns with excessive nulls: {excessive_nulls.to_dict()}")
            self.validation_results['excessive_nulls'] = excessive_nulls.to_dict()
            return False
        
        return True
    
    def validate_price_consistency(self) -> bool:
        """Check price column consistency."""
        if all(col in self.data.columns for col in ['High', 'Low', 'Close']):
            # High should be >= Low
            invalid_rows = self.data[self.data['High'] < self.data['Low']]
            if not invalid_rows.empty:
                logger.error(f"Found {len(invalid_rows)} rows where High < Low")
                return False
        
        return True
    
    def get_validation_report(self) -> Dict:
        """Return validation results."""
        return self.validation_results
