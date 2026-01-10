"""
Data cleaning module
"""
import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any
from src.utils.logger import get_logger
from src.utils.config import ConfigManager

logger = get_logger()


class DataCleaner:
    """Clean and prepare data"""
    
    def __init__(self, config: Optional[ConfigManager] = None):
        """
        Initialize data cleaner
        
        Args:
            config: Configuration manager instance
        """
        self.config = config or ConfigManager()
        self.cleaning_report = {
            'actions': [],
            'statistics': {}
        }
    
    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Perform comprehensive data cleaning
        
        Args:
            df: DataFrame to clean
        
        Returns:
            Cleaned DataFrame
        """
        logger.info("Starting data cleaning...")
        df_clean = df.copy()
        
        # Clean column names
        df_clean = self._clean_column_names(df_clean)
        
        # Handle missing values
        df_clean = self._handle_missing_values(df_clean)
        
        # Remove duplicates
        df_clean = self._remove_duplicates(df_clean)
        
        # Handle outliers
        df_clean = self._handle_outliers(df_clean)
        
        # Fix data types
        df_clean = self._fix_data_types(df_clean)
        
        # Clean text columns
        df_clean = self._clean_text_columns(df_clean)
        
        logger.info(f"Cleaning complete. Shape: {df.shape} -> {df_clean.shape}")
        return df_clean
    
    def _clean_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize column names"""
        original_cols = df.columns.tolist()
        
        # Convert to lowercase, replace spaces with underscores
        df.columns = (df.columns
                      .str.lower()
                      .str.replace(' ', '_')
                      .str.replace('[^a-z0-9_]', '', regex=True))
        
        # Handle duplicate column names
        cols = pd.Series(df.columns)
        for dup in cols[cols.duplicated()].unique():
            cols[cols[cols == dup].index.values.tolist()] = [
                dup + '_' + str(i) if i != 0 else dup 
                for i in range(sum(cols == dup))
            ]
        df.columns = cols
        
        if original_cols != df.columns.tolist():
            self.cleaning_report['actions'].append("Cleaned column names")
            logger.info("Column names cleaned and standardized")
        
        return df
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values based on configuration"""
        strategy = self.config.get('cleaning.missing_values.strategy', 'auto')
        threshold = self.config.get('cleaning.missing_values.threshold', 0.5)
        
        initial_shape = df.shape
        missing_before = df.isna().sum().sum()
        
        # Drop columns with too many missing values
        missing_pct = df.isna().sum() / len(df)
        cols_to_drop = missing_pct[missing_pct > threshold].index.tolist()
        
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)
            logger.info(f"Dropped {len(cols_to_drop)} columns with >{threshold*100}% missing: {cols_to_drop}")
            self.cleaning_report['actions'].append(f"Dropped {len(cols_to_drop)} high-missing columns")
        
        # Handle remaining missing values
        if strategy == 'auto':
            df = self._auto_fill_missing(df)
        elif strategy == 'drop':
            df = df.dropna()
        elif strategy == 'fill_mean':
            df = df.fillna(df.mean(numeric_only=True))
        elif strategy == 'fill_median':
            df = df.fillna(df.median(numeric_only=True))
        elif strategy == 'fill_forward':
            df = df.fillna(method='ffill')
        elif strategy == 'fill_backward':
            df = df.fillna(method='bfill')
        
        missing_after = df.isna().sum().sum()
        if missing_after < missing_before:
            self.cleaning_report['statistics']['missing_values_handled'] = int(missing_before - missing_after)
            logger.info(f"Handled {missing_before - missing_after} missing values")
        
        return df
    
    def _auto_fill_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """Automatically fill missing values based on column type"""
        for col in df.columns:
            if df[col].isna().sum() == 0:
                continue
            
            # Numeric columns: fill with median
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col].fillna(df[col].median(), inplace=True)
            
            # Categorical/Object columns: fill with mode
            elif pd.api.types.is_object_dtype(df[col]):
                mode_value = df[col].mode()
                if len(mode_value) > 0:
                    df[col].fillna(mode_value[0], inplace=True)
                else:
                    df[col].fillna('Unknown', inplace=True)
            
            # DateTime columns: forward fill
            elif pd.api.types.is_datetime64_any_dtype(df[col]):
                df[col].fillna(method='ffill', inplace=True)
        
        return df
    
    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate rows"""
        keep = self.config.get('cleaning.duplicates.keep', 'first')
        subset = self.config.get('cleaning.duplicates.subset')
        
        initial_rows = len(df)
        df = df.drop_duplicates(subset=subset, keep=keep)
        
        duplicates_removed = initial_rows - len(df)
        if duplicates_removed > 0:
            self.cleaning_report['statistics']['duplicates_removed'] = duplicates_removed
            logger.info(f"Removed {duplicates_removed} duplicate rows")
        
        return df
    
    def _handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect and handle outliers in numeric columns"""
        method = self.config.get('cleaning.outliers.method', 'iqr')
        threshold = self.config.get('cleaning.outliers.threshold', 1.5)
        action = self.config.get('cleaning.outliers.action', 'cap')
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        outliers_handled = 0
        
        for col in numeric_cols:
            if method == 'iqr':
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                
                outlier_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
                n_outliers = outlier_mask.sum()
                
                if n_outliers > 0:
                    if action == 'cap':
                        df.loc[df[col] < lower_bound, col] = lower_bound
                        df.loc[df[col] > upper_bound, col] = upper_bound
                    elif action == 'remove':
                        df = df[~outlier_mask]
                    elif action == 'flag':
                        df[f'{col}_is_outlier'] = outlier_mask
                    
                    outliers_handled += n_outliers
            
            elif method == 'zscore':
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                outlier_mask = z_scores > threshold
                n_outliers = outlier_mask.sum()
                
                if n_outliers > 0 and action != 'flag':
                    if action == 'cap':
                        mean = df[col].mean()
                        std = df[col].std()
                        df.loc[outlier_mask, col] = mean + threshold * std * np.sign(df.loc[outlier_mask, col] - mean)
                    elif action == 'remove':
                        df = df[~outlier_mask]
                    
                    outliers_handled += n_outliers
        
        if outliers_handled > 0:
            self.cleaning_report['statistics']['outliers_handled'] = outliers_handled
            logger.info(f"Handled {outliers_handled} outliers using {method} method")
        
        return df
    
    def _fix_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Automatically fix data types"""
        for col in df.columns:
            # Try to convert object columns to numeric
            if df[col].dtype == 'object':
                try:
                    df[col] = pd.to_numeric(df[col])
                    logger.debug(f"Converted '{col}' to numeric")
                except:
                    pass
                
                # Try to convert to datetime
                try:
                    if df[col].astype(str).str.match(r'\d{4}-\d{2}-\d{2}').any():
                        df[col] = pd.to_datetime(df[col], errors='coerce')
                        logger.debug(f"Converted '{col}' to datetime")
                except:
                    pass
        
        return df
    
    def _clean_text_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean text columns"""
        text_cols = df.select_dtypes(include=['object']).columns
        
        for col in text_cols:
            # Strip whitespace
            df[col] = df[col].astype(str).str.strip()
            
            # Replace multiple spaces with single space
            df[col] = df[col].str.replace(r'\s+', ' ', regex=True)
            
            # Remove special characters if needed (optional)
            # df[col] = df[col].str.replace(r'[^\w\s]', '', regex=True)
        
        if len(text_cols) > 0:
            logger.info(f"Cleaned {len(text_cols)} text columns")
        
        return df
    
    def get_report(self) -> Dict[str, Any]:
        """Get cleaning report"""
        return self.cleaning_report