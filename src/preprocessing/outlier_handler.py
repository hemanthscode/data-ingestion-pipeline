"""
Outlier detection and handling module
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.ensemble import IsolationForest
from scipy import stats
from src.utils.logger import get_logger
from src.utils.config import ConfigManager

logger = get_logger()


class OutlierHandler:
    """Detect and handle outliers in numerical data"""
    
    def __init__(self, config: Optional[ConfigManager] = None):
        """
        Initialize outlier handler
        
        Args:
            config: Configuration manager instance
        """
        self.config = config or ConfigManager()
        self.outlier_info = {}
    
    def detect_and_handle(self, 
                         df: pd.DataFrame, 
                         method: str = 'iqr',
                         action: str = 'cap',
                         threshold: float = 1.5) -> pd.DataFrame:
        """
        Detect and handle outliers
        
        Args:
            df: DataFrame to process
            method: Detection method ('iqr', 'zscore', 'isolation_forest', 'modified_zscore')
            action: Action to take ('cap', 'remove', 'flag', 'winsorize')
            threshold: Threshold for outlier detection
        
        Returns:
            DataFrame with outliers handled
        """
        logger.info(f"Detecting outliers using {method} method...")
        
        df_processed = df.copy()
        numeric_cols = df_processed.select_dtypes(include=[np.number]).columns.tolist()
        
        # Exclude binary columns
        binary_cols = [col for col in numeric_cols 
                      if df_processed[col].nunique() == 2 
                      and set(df_processed[col].unique()).issubset({0, 1})]
        cols_to_check = [col for col in numeric_cols if col not in binary_cols]
        
        if not cols_to_check:
            logger.info("No numeric columns to check for outliers")
            return df_processed
        
        # Detect outliers based on method
        if method == 'iqr':
            df_processed = self._handle_iqr_outliers(df_processed, cols_to_check, action, threshold)
        elif method == 'zscore':
            df_processed = self._handle_zscore_outliers(df_processed, cols_to_check, action, threshold)
        elif method == 'modified_zscore':
            df_processed = self._handle_modified_zscore_outliers(df_processed, cols_to_check, action, threshold)
        elif method == 'isolation_forest':
            df_processed = self._handle_isolation_forest_outliers(df_processed, cols_to_check, action)
        else:
            logger.warning(f"Unknown method: {method}. Using IQR instead.")
            df_processed = self._handle_iqr_outliers(df_processed, cols_to_check, action, threshold)
        
        # Log summary
        total_outliers = sum(info['count'] for info in self.outlier_info.values())
        logger.info(f"Found {total_outliers} outliers across {len(self.outlier_info)} columns")
        
        return df_processed
    
    def _handle_iqr_outliers(self, 
                            df: pd.DataFrame, 
                            columns: List[str], 
                            action: str, 
                            threshold: float) -> pd.DataFrame:
        """Handle outliers using IQR method"""
        for col in columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            
            outlier_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
            n_outliers = outlier_mask.sum()
            
            if n_outliers > 0:
                self.outlier_info[col] = {
                    'method': 'iqr',
                    'count': int(n_outliers),
                    'percentage': round((n_outliers / len(df)) * 100, 2),
                    'lower_bound': round(lower_bound, 4),
                    'upper_bound': round(upper_bound, 4)
                }
                
                # Apply action
                df = self._apply_action(df, col, outlier_mask, lower_bound, upper_bound, action)
        
        return df
    
    def _handle_zscore_outliers(self, 
                                df: pd.DataFrame, 
                                columns: List[str], 
                                action: str, 
                                threshold: float) -> pd.DataFrame:
        """Handle outliers using Z-score method"""
        for col in columns:
            mean = df[col].mean()
            std = df[col].std()
            
            if std == 0:
                continue
            
            z_scores = np.abs((df[col] - mean) / std)
            outlier_mask = z_scores > threshold
            n_outliers = outlier_mask.sum()
            
            if n_outliers > 0:
                lower_bound = mean - threshold * std
                upper_bound = mean + threshold * std
                
                self.outlier_info[col] = {
                    'method': 'zscore',
                    'count': int(n_outliers),
                    'percentage': round((n_outliers / len(df)) * 100, 2),
                    'lower_bound': round(lower_bound, 4),
                    'upper_bound': round(upper_bound, 4),
                    'threshold_zscore': threshold
                }
                
                # Apply action
                df = self._apply_action(df, col, outlier_mask, lower_bound, upper_bound, action)
        
        return df
    
    def _handle_modified_zscore_outliers(self, 
                                        df: pd.DataFrame, 
                                        columns: List[str], 
                                        action: str, 
                                        threshold: float = 3.5) -> pd.DataFrame:
        """Handle outliers using Modified Z-score (using median)"""
        for col in columns:
            median = df[col].median()
            mad = np.median(np.abs(df[col] - median))
            
            if mad == 0:
                continue
            
            modified_z_scores = 0.6745 * (df[col] - median) / mad
            outlier_mask = np.abs(modified_z_scores) > threshold
            n_outliers = outlier_mask.sum()
            
            if n_outliers > 0:
                self.outlier_info[col] = {
                    'method': 'modified_zscore',
                    'count': int(n_outliers),
                    'percentage': round((n_outliers / len(df)) * 100, 2),
                    'median': round(median, 4),
                    'mad': round(mad, 4)
                }
                
                # Calculate bounds for capping
                lower_bound = median - threshold * mad / 0.6745
                upper_bound = median + threshold * mad / 0.6745
                
                # Apply action
                df = self._apply_action(df, col, outlier_mask, lower_bound, upper_bound, action)
        
        return df
    
    def _handle_isolation_forest_outliers(self, 
                                         df: pd.DataFrame, 
                                         columns: List[str], 
                                         action: str) -> pd.DataFrame:
        """Handle outliers using Isolation Forest"""
        if len(columns) == 0:
            return df
        
        # Use Isolation Forest on all numeric columns together
        iso_forest = IsolationForest(contamination=0.1, random_state=42)
        
        try:
            predictions = iso_forest.fit_predict(df[columns])
            outlier_mask = predictions == -1
            n_outliers = outlier_mask.sum()
            
            if n_outliers > 0:
                self.outlier_info['__multivariate__'] = {
                    'method': 'isolation_forest',
                    'count': int(n_outliers),
                    'percentage': round((n_outliers / len(df)) * 100, 2),
                    'columns': columns
                }
                
                if action == 'remove':
                    df = df[~outlier_mask]
                    logger.info(f"Removed {n_outliers} multivariate outliers")
                elif action == 'flag':
                    df['is_outlier_multivariate'] = outlier_mask.astype(int)
                    logger.info(f"Flagged {n_outliers} multivariate outliers")
        
        except Exception as e:
            logger.error(f"Isolation Forest failed: {e}")
        
        return df
    
    def _apply_action(self, 
                     df: pd.DataFrame, 
                     col: str, 
                     outlier_mask: pd.Series, 
                     lower_bound: float, 
                     upper_bound: float, 
                     action: str) -> pd.DataFrame:
        """Apply the specified action to outliers"""
        if action == 'cap':
            # Cap outliers to bounds
            df.loc[df[col] < lower_bound, col] = lower_bound
            df.loc[df[col] > upper_bound, col] = upper_bound
            logger.debug(f"Capped {outlier_mask.sum()} outliers in '{col}'")
        
        elif action == 'remove':
            # Remove rows with outliers
            df = df[~outlier_mask]
            logger.debug(f"Removed {outlier_mask.sum()} outliers from '{col}'")
        
        elif action == 'flag':
            # Add flag column
            df[f'{col}_is_outlier'] = outlier_mask.astype(int)
            logger.debug(f"Flagged {outlier_mask.sum()} outliers in '{col}'")
        
        elif action == 'winsorize':
            # Winsorize (similar to cap but based on percentiles)
            lower_percentile = df[col].quantile(0.01)
            upper_percentile = df[col].quantile(0.99)
            df.loc[df[col] < lower_percentile, col] = lower_percentile
            df.loc[df[col] > upper_percentile, col] = upper_percentile
            logger.debug(f"Winsorized {outlier_mask.sum()} outliers in '{col}'")
        
        return df
    
    def get_outlier_summary(self) -> Dict[str, any]:
        """Get summary of detected outliers"""
        return {
            'total_columns_with_outliers': len(self.outlier_info),
            'details': self.outlier_info
        }
    
    def visualize_outliers(self, df: pd.DataFrame, column: str) -> Dict[str, any]:
        """
        Get outlier visualization data for a specific column
        
        Args:
            df: DataFrame
            column: Column name
        
        Returns:
            Dictionary with visualization data
        """
        if column not in df.columns or not pd.api.types.is_numeric_dtype(df[column]):
            return {'error': f"Column '{column}' not found or not numeric"}
        
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)][column]
        
        return {
            'column': column,
            'n_outliers': len(outliers),
            'outlier_percentage': round((len(outliers) / len(df)) * 100, 2),
            'bounds': {
                'lower': round(lower_bound, 4),
                'upper': round(upper_bound, 4)
            },
            'statistics': {
                'min': df[column].min(),
                'q1': Q1,
                'median': df[column].median(),
                'q3': Q3,
                'max': df[column].max(),
                'mean': df[column].mean(),
                'std': df[column].std()
            },
            'outlier_values': outliers.tolist()[:10]  # First 10 outlier values
        }