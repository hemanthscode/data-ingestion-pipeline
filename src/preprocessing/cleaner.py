"""
Data cleaning module (FULLY FIXED)
"""
import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any
from src.utils.logger import get_logger
from src.utils.config import ConfigManager

logger = get_logger()

class DataCleaner:
    """Clean and prepare data - BUSINESS SAFE"""
    
    def __init__(self, config: Optional[ConfigManager] = None):
        self.config = config or ConfigManager()
        self.cleaning_report = {
            'actions': [],
            'statistics': {}
        }
    
    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Perform comprehensive data cleaning"""
        logger.info("🧹 Starting data cleaning...")
        df_clean = df.copy()
        
        # Clean column names
        df_clean = self._clean_column_names(df_clean)
        
        # Handle missing values
        df_clean = self._handle_missing_values(df_clean)
        
        # Remove duplicates
        df_clean = self._remove_duplicates(df_clean)
        
        # Handle outliers
        df_clean = self._handle_outliers(df_clean)
        
        # Fix data types (BUSINESS SAFE)
        df_clean = self._fix_data_types(df_clean)
        
        # Clean text columns
        df_clean = self._clean_text_columns(df_clean)
        
        logger.info(f"✅ Cleaning complete. Shape: {df.shape} -> {df_clean.shape}")
        self.cleaning_report['actions'].append('Full cleaning pipeline')
        return df_clean
    
    def _clean_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize column names"""
        original_cols = df.columns.tolist()
        
        df.columns = (df.columns
                      .str.lower()
                      .str.replace(' ', '_')
                      .str.replace('[^a-z0-9_]', '', regex=True))
        
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
        strategy = self.config.get('cleaning.missing_values.strategy', 'fill_median')
        threshold = self.config.get('cleaning.missing_values.threshold', 0.5)
        
        initial_shape = df.shape
        missing_before = df.isna().sum().sum()
        
        # Drop columns with too many missing values
        missing_pct = df.isna().sum() / len(df)
        cols_to_drop = missing_pct[missing_pct > threshold].index.tolist()
        
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)
            logger.info(f"Dropped {len(cols_to_drop)} columns with >{threshold*100}% missing")
            self.cleaning_report['actions'].append(f"Dropped {len(cols_to_drop)} high-missing columns")
        
        # Handle remaining missing values
        if strategy == 'fill_median':
            df = df.fillna(df.median(numeric_only=True))
        elif strategy == 'auto':
            df = self._auto_fill_missing(df)
        
        missing_after = df.isna().sum().sum()
        if missing_after < missing_before:
            self.cleaning_report['statistics']['missing_values_handled'] = int(missing_before - missing_after)
            logger.info(f"✅ Handled {missing_before - missing_after} missing values")
        
        return df
    
    def _auto_fill_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """Auto fill missing values by column type"""
        for col in df.columns:
            if df[col].isna().sum() == 0:
                continue
            
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col].fillna(df[col].median(), inplace=True)
            elif pd.api.types.is_object_dtype(df[col]):
                mode_value = df[col].mode()
                if len(mode_value) > 0:
                    df[col].fillna(mode_value[0], inplace=True)
                else:
                    df[col].fillna('Unknown', inplace=True)
        return df
    
    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate rows"""
        keep = self.config.get('cleaning.duplicates.keep', 'first')
        initial_rows = len(df)
        df = df.drop_duplicates(keep=keep)
        
        duplicates_removed = initial_rows - len(df)
        if duplicates_removed > 0:
            self.cleaning_report['statistics']['duplicates_removed'] = duplicates_removed
            logger.info(f"Removed {duplicates_removed} duplicate rows")
        return df
    
    def _handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle outliers - BUSINESS SAFE (cap only)"""
        method = self.config.get('cleaning.outliers.method', 'iqr')
        threshold = self.config.get('cleaning.outliers.threshold', 1.5)
        action = self.config.get('cleaning.outliers.action', 'cap')
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        outliers_handled = 0
        
        # BUSINESS COLUMNS - CAP OUTLIERS (never remove rows)
        business_cols = ['age', 'salary']
        for col in numeric_cols:
            if col in business_cols and action == 'cap':
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                
                outlier_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
                n_outliers = outlier_mask.sum()
                
                if n_outliers > 0:
                    df.loc[df[col] < lower_bound, col] = lower_bound
                    df.loc[df[col] > upper_bound, col] = upper_bound
                    outliers_handled += n_outliers
                    logger.info(f"🔒 Capped {n_outliers} outliers in {col}")
        
        if outliers_handled > 0:
            self.cleaning_report['statistics']['outliers_handled'] = outliers_handled
        return df
    
    def _fix_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """FIXED: Business-safe data type correction"""
        business_columns = ['age', 'salary', 'phone', 'employee_id']
        
        for col in df.columns:
            # PROTECT BUSINESS COLUMNS
            if col.lower() in [b.lower() for b in business_columns]:
                if 'phone' in col.lower():
                    df[col] = df[col].astype(str)
                    logger.info(f"📱 Phone column protected as string: {col}")
                continue
            
            # Convert object → numeric (safe columns only)
            if df[col].dtype == 'object' and 'phone' not in col.lower():
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                except:
                    pass
            
            # Date columns
            try:
                if df[col].dtype == 'object':
                    temp_date = pd.to_datetime(df[col], errors='coerce')
                    if temp_date.notna().sum() > len(df) * 0.5:
                        df[col] = temp_date
                        logger.info(f"📅 Converted {col} to datetime")
            except:
                pass
        
        return df
    
    def _clean_text_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean text columns"""
        text_cols = df.select_dtypes(include=['object']).columns
        
        for col in text_cols:
            df[col] = df[col].astype(str).str.strip()
            df[col] = df[col].str.replace(r'\s+', ' ', regex=True)
        
        logger.info(f"Cleaned {len(text_cols)} text columns")
        return df
    
    def get_report(self) -> Dict[str, Any]:
        return self.cleaning_report
