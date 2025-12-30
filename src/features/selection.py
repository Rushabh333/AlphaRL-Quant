"""
Feature selection module.
"""
import pandas as pd
from typing import List
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from src.utils.logger import get_logger

logger = get_logger(__name__)


class FeatureSelector:
    """
    Select most important features for modeling.
    """
    
    def __init__(self, data: pd.DataFrame, target_column: str = None):
        self.data = data
        self.target_column = target_column
        self.selected_features = []
    
    def select_by_correlation(self, threshold: float = 0.95) -> pd.DataFrame:
        """Remove highly correlated features."""
        logger.info(f"Selecting features with correlation threshold: {threshold}")
        
        # Calculate correlation matrix
        corr_matrix = self.data.corr().abs()
        
        # Select upper triangle of correlation matrix
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        
        # Find features with correlation greater than threshold
        to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
        
        logger.info(f"Dropping {len(to_drop)} highly correlated features")
        
        return self.data.drop(columns=to_drop)
    
    def select_k_best(self, k: int = 10) -> List[str]:
        """Select K best features."""
        if self.target_column is None:
            raise ValueError("Target column must be specified for feature selection")
        
        logger.info(f"Selecting {k} best features")
        
        X = self.data.drop(columns=[self.target_column])
        y = self.data[self.target_column]
        
        selector = SelectKBest(score_func=f_classif, k=k)
        selector.fit(X, y)
        
        selected_features = X.columns[selector.get_support()].tolist()
        self.selected_features = selected_features
        
        logger.info(f"Selected features: {selected_features}")
        return selected_features
