"""
Data profiling and quality assessment module
"""
import pandas as pd
import numpy as np
from typing import Dict, Any
from src.utils.logger import get_logger

logger = get_logger()

class DataProfiler:
    """Generate comprehensive data profiles"""
    
    def __init__(self):
        self.profile = {}
    
    def generate_profile(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate comprehensive data profile
        
        Args:
            df: DataFrame to profile
            
        Returns:
            Dictionary containing profile information
        """
        logger.info("Generating data profile...")
        
        self.profile = {
            'overview': self._get_overview(df),
            'columns': self._profile_columns(df),
            'quality_metrics': self._calculate_quality_metrics(df),
            'correlations': self._get_correlations(df)
        }
        
        logger.info("Data profiling complete")
        return self.profile
    
    def _get_overview(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get dataset overview"""
        return {
            'n_rows': len(df),
            'n_columns': len(df.columns),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2,
            'duplicate_rows': df.duplicated().sum(),
            'duplicate_percentage': (df.duplicated().sum() / len(df)) * 100
        }
    
    def _profile_columns(self, df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """Profile each column - FIXED for boolean dtype compatibility"""
        column_profiles = {}
        
        for col in df.columns:
            profile = {
                'dtype': str(df[col].dtype),
                'n_unique': df[col].nunique(),
                'n_missing': df[col].isna().sum(),
                'missing_percentage': (df[col].isna().sum() / len(df)) * 100
            }
            
            # FIXED: Check for strictly numeric dtypes only (excludes bool)
            if pd.api.types.is_numeric_dtype(df[col]) and str(df[col].dtype) not in ['bool', 'boolean']:
                try:
                    # Safe numeric profiling with error handling
                    profile.update({
                        'mean': float(df[col].mean()) if df[col].count() > 0 else None,
                        'median': float(df[col].median()) if df[col].count() > 0 else None,
                        'std': float(df[col].std()) if df[col].count() > 0 else None,
                        'min': float(df[col].min()) if df[col].count() > 0 else None,
                        'max': float(df[col].max()) if df[col].count() > 0 else None,
                        'q25': float(df[col].quantile(0.25)) if df[col].count() > 0 else None,
                        'q75': float(df[col].quantile(0.75)) if df[col].count() > 0 else None,
                        'skewness': float(df[col].skew()) if df[col].count() > 5 else None,
                        'kurtosis': float(df[col].kurtosis()) if df[col].count() > 5 else None
                    })
                except Exception as e:
                    logger.warning(f"Could not compute numeric stats for {col}: {str(e)}")
                    # Fallback to basic numeric info only
                    profile.update({
                        'mean': None, 'median': None, 'std': None,
                        'min': None, 'max': None, 'q25': None, 'q75': None,
                        'skewness': None, 'kurtosis': None
                    })
            
            # Categorical/Object columns
            elif pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_categorical_dtype(df[col]):
                value_counts = df[col].value_counts()
                profile.update({
                    'most_common': value_counts.index[0] if len(value_counts) > 0 else None,
                    'most_common_count': int(value_counts.iloc[0]) if len(value_counts) > 0 else 0,
                    'top_5_values': value_counts.head(5).to_dict()
                })
            
            # Boolean columns - explicit handling
            elif str(df[col].dtype) in ['bool', 'boolean']:
                value_counts = df[col].value_counts()
                profile.update({
                    'true_count': int(value_counts.get(True, 0)),
                    'false_count': int(value_counts.get(False, 0)),
                    'true_percentage': (value_counts.get(True, 0) / len(df)) * 100,
                    'top_5_values': value_counts.head(5).to_dict()
                })
            
            # DateTime columns
            elif pd.api.types.is_datetime64_any_dtype(df[col]):
                profile.update({
                    'min_date': df[col].min(),
                    'max_date': df[col].max(),
                    'date_range_days': (df[col].max() - df[col].min()).days if df[col].count() > 1 else 0
                })
            
            column_profiles[col] = profile
        
        return column_profiles
    
    def _calculate_quality_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate overall data quality metrics"""
        total_cells = df.shape[0] * df.shape[1]
        missing_cells = df.isna().sum().sum()
        
        return {
            'completeness': ((total_cells - missing_cells) / total_cells) * 100,
            'total_missing_values': int(missing_cells),
            'columns_with_missing': int((df.isna().sum() > 0).sum()),
            'fully_complete_columns': int((df.isna().sum() == 0).sum())
        }
    
    def _get_correlations(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate correlations for numeric columns only"""
        # Exclude boolean columns explicitly
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_df = df[[col for col in numeric_cols if str(df[col].dtype) not in ['bool', 'boolean']]]
        
        if numeric_df.shape[1] < 2:
            return {'message': 'Not enough numeric columns for correlation'}
        
        try:
            corr_matrix = numeric_df.corr()
            
            # Find highly correlated pairs (excluding diagonal)
            high_corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    if abs(corr_matrix.iloc[i, j]) > 0.8:
                        high_corr_pairs.append({
                            'col1': corr_matrix.columns[i],
                            'col2': corr_matrix.columns[j],
                            'correlation': round(float(corr_matrix.iloc[i, j]), 3)
                        })
            
            return {
                'high_correlations': high_corr_pairs,
                'n_high_correlations': len(high_corr_pairs)
            }
        except Exception as e:
            logger.warning(f"Correlation calculation failed: {str(e)}")
            return {'message': f'Correlation failed: {str(e)}', 'n_high_correlations': 0}
    
    def print_summary(self):
        """Print a human-readable summary"""
        if not self.profile:
            print("No profile generated yet")
            return
        
        print("=" * 60)
        print("DATA PROFILE SUMMARY")
        print("=" * 60)
        
        # Overview
        print(f"\n📊 OVERVIEW:")
        print(f"  Rows: {self.profile['overview']['n_rows']:,}")
        print(f"  Columns: {self.profile['overview']['n_columns']}")
        print(f"  Memory: {self.profile['overview']['memory_usage_mb']:.2f} MB")
        print(f"  Duplicates: {self.profile['overview']['duplicate_rows']:,} ({self.profile['overview']['duplicate_percentage']:.2f}%)")
        
        # Quality
        print(f"\n✨ QUALITY METRICS:")
        print(f"  Completeness: {self.profile['quality_metrics']['completeness']:.2f}%")
        print(f"  Missing Values: {self.profile['quality_metrics']['total_missing_values']:,}")
        print(f"  Columns with Missing: {self.profile['quality_metrics']['columns_with_missing']}/{self.profile['overview']['n_columns']}")
        
        # Correlations
        if 'high_correlations' in self.profile['correlations']:
            n_high = self.profile['correlations']['n_high_correlations']
            print(f"\n🔗 HIGH CORRELATIONS: {n_high} pairs with |r| > 0.8")
        
        print("=" * 60)
