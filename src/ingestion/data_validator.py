"""
Data validation module
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from src.utils.logger import get_logger

logger = get_logger()


class DataValidator:
    """Validate data quality and structure"""
    
    def __init__(self):
        self.validation_results = {}
    
    def validate(self, df: pd.DataFrame, schema: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Perform comprehensive data validation
        
        Args:
            df: DataFrame to validate
            schema: Optional schema definition
        
        Returns:
            Validation results dictionary
        """
        logger.info("Starting data validation...")
        
        self.validation_results = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'info': {}
        }
        
        # Basic checks
        self._check_empty(df)
        self._check_columns(df)
        self._check_data_types(df)
        self._check_missing_values(df)
        self._check_duplicates(df)
        self._check_outliers(df)
        
        # Schema validation if provided
        if schema:
            self._validate_schema(df, schema)
        
        # Summary
        self.validation_results['info']['total_rows'] = len(df)
        self.validation_results['info']['total_columns'] = len(df.columns)
        
        if self.validation_results['errors']:
            self.validation_results['is_valid'] = False
            logger.warning(f"Validation failed with {len(self.validation_results['errors'])} errors")
        else:
            logger.info("Validation passed")
        
        return self.validation_results
    
    def _check_empty(self, df: pd.DataFrame):
        """Check if DataFrame is empty"""
        if df.empty:
            self.validation_results['errors'].append("DataFrame is empty")
            logger.error("DataFrame is empty")
    
    def _check_columns(self, df: pd.DataFrame):
        """Check column names"""
        # Check for unnamed columns
        unnamed_cols = [col for col in df.columns if 'Unnamed' in str(col)]
        if unnamed_cols:
            self.validation_results['warnings'].append(f"Found {len(unnamed_cols)} unnamed columns")
            logger.warning(f"Unnamed columns: {unnamed_cols}")
        
        # Check for duplicate column names
        duplicates = df.columns[df.columns.duplicated()].tolist()
        if duplicates:
            self.validation_results['errors'].append(f"Duplicate column names: {duplicates}")
            logger.error(f"Duplicate columns: {duplicates}")
    
    def _check_data_types(self, df: pd.DataFrame):
        """Check and infer data types"""
        type_info = {}
        
        for col in df.columns:
            dtype = str(df[col].dtype)
            type_info[col] = dtype
            
            # Check if object type could be converted
            if dtype == 'object':
                # Try to infer better type
                try:
                    pd.to_numeric(df[col])
                    self.validation_results['warnings'].append(
                        f"Column '{col}' is object but could be numeric"
                    )
                except:
                    pass
        
        self.validation_results['info']['data_types'] = type_info
    
    def _check_missing_values(self, df: pd.DataFrame):
        """Check for missing values"""
        missing_info = {}
        
        for col in df.columns:
            missing_count = df[col].isna().sum()
            missing_pct = (missing_count / len(df)) * 100
            
            if missing_count > 0:
                missing_info[col] = {
                    'count': int(missing_count),
                    'percentage': round(missing_pct, 2)
                }
                
                if missing_pct > 50:
                    self.validation_results['warnings'].append(
                        f"Column '{col}' has {missing_pct:.1f}% missing values"
                    )
        
        if missing_info:
            self.validation_results['info']['missing_values'] = missing_info
            logger.info(f"Found missing values in {len(missing_info)} columns")
    
    def _check_duplicates(self, df: pd.DataFrame):
        """Check for duplicate rows"""
        duplicates = df.duplicated().sum()
        duplicate_pct = (duplicates / len(df)) * 100
        
        if duplicates > 0:
            self.validation_results['info']['duplicates'] = {
                'count': int(duplicates),
                'percentage': round(duplicate_pct, 2)
            }
            
            if duplicate_pct > 10:
                self.validation_results['warnings'].append(
                    f"High duplicate ratio: {duplicate_pct:.1f}%"
                )
            
            logger.info(f"Found {duplicates} duplicate rows ({duplicate_pct:.2f}%)")
    
    def _check_outliers(self, df: pd.DataFrame):
        """Detect outliers in numeric columns using IQR method"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        outlier_info = {}
        
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
            
            if len(outliers) > 0:
                outlier_pct = (len(outliers) / len(df)) * 100
                outlier_info[col] = {
                    'count': len(outliers),
                    'percentage': round(outlier_pct, 2),
                    'lower_bound': round(lower_bound, 2),
                    'upper_bound': round(upper_bound, 2)
                }
        
        if outlier_info:
            self.validation_results['info']['outliers'] = outlier_info
            logger.info(f"Detected outliers in {len(outlier_info)} numeric columns")
    
    def _validate_schema(self, df: pd.DataFrame, schema: Dict):
        """
        Validate DataFrame against schema
        
        Schema format:
        {
            'columns': ['col1', 'col2'],
            'required_columns': ['col1'],
            'dtypes': {'col1': 'int64', 'col2': 'object'}
        }
        """
        # Check required columns
        if 'required_columns' in schema:
            missing_cols = set(schema['required_columns']) - set(df.columns)
            if missing_cols:
                self.validation_results['errors'].append(
                    f"Missing required columns: {missing_cols}"
                )
        
        # Check data types
        if 'dtypes' in schema:
            for col, expected_type in schema['dtypes'].items():
                if col in df.columns:
                    actual_type = str(df[col].dtype)
                    if actual_type != expected_type:
                        self.validation_results['warnings'].append(
                            f"Column '{col}': expected {expected_type}, got {actual_type}"
                        )
    
    def get_summary(self) -> str:
        """Get validation summary as string"""
        summary = []
        summary.append("=" * 50)
        summary.append("DATA VALIDATION SUMMARY")
        summary.append("=" * 50)
        
        if self.validation_results['is_valid']:
            summary.append("✓ Validation PASSED")
        else:
            summary.append("✗ Validation FAILED")
        
        if self.validation_results['errors']:
            summary.append(f"\nErrors ({len(self.validation_results['errors'])}):")
            for error in self.validation_results['errors']:
                summary.append(f"  - {error}")
        
        if self.validation_results['warnings']:
            summary.append(f"\nWarnings ({len(self.validation_results['warnings'])}):")
            for warning in self.validation_results['warnings']:
                summary.append(f"  - {warning}")
        
        summary.append("=" * 50)
        return "\n".join(summary)