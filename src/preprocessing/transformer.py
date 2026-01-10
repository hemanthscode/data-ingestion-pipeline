"""
Data transformation module for encoding and scaling
"""
import pandas as pd
import numpy as np
from typing import Optional, Dict, List, Any
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from src.utils.logger import get_logger
from src.utils.config import ConfigManager

logger = get_logger()


class DataTransformer:
    """Transform data for ML/analysis"""
    
    def __init__(self, config: Optional[ConfigManager] = None):
        """
        Initialize data transformer
        
        Args:
            config: Configuration manager instance
        """
        self.config = config or ConfigManager()
        self.encoders = {}
        self.scalers = {}
        self.transformation_report = {
            'encoded_columns': [],
            'scaled_columns': [],
            'engineered_features': []
        }
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Perform complete transformation pipeline
        
        Args:
            df: DataFrame to transform
        
        Returns:
            Transformed DataFrame
        """
        logger.info("Starting data transformation...")
        df_transformed = df.copy()
        
        # Feature engineering
        df_transformed = self._engineer_features(df_transformed)
        
        # Encode categorical variables
        df_transformed = self._encode_categorical(df_transformed)
        
        # Scale numerical features
        df_transformed = self._scale_numerical(df_transformed)
        
        logger.info("Transformation complete")
        return df_transformed
    
    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create new features from existing data"""
        create_datetime = self.config.get('transformation.feature_engineering.create_datetime_features', True)
        
        if create_datetime:
            df = self._create_datetime_features(df)
        
        return df
    
    def _create_datetime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract features from datetime columns"""
        datetime_cols = df.select_dtypes(include=['datetime64']).columns
        
        for col in datetime_cols:
            # Extract year, month, day, etc.
            df[f'{col}_year'] = df[col].dt.year
            df[f'{col}_month'] = df[col].dt.month
            df[f'{col}_day'] = df[col].dt.day
            df[f'{col}_dayofweek'] = df[col].dt.dayofweek
            df[f'{col}_quarter'] = df[col].dt.quarter
            df[f'{col}_is_weekend'] = (df[col].dt.dayofweek >= 5).astype(int)
            
            self.transformation_report['engineered_features'].extend([
                f'{col}_year', f'{col}_month', f'{col}_day', 
                f'{col}_dayofweek', f'{col}_quarter', f'{col}_is_weekend'
            ])
            
            logger.info(f"Created datetime features from '{col}'")
        
        return df
    
    def _encode_categorical(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical variables"""
        method = self.config.get('transformation.categorical_encoding.method', 'auto')
        max_categories = self.config.get('transformation.categorical_encoding.max_categories', 50)
        
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        for col in categorical_cols:
            n_unique = df[col].nunique()
            
            # Auto-select encoding method
            if method == 'auto':
                if n_unique == 2:
                    encoding_method = 'label'
                elif n_unique <= 10:
                    encoding_method = 'onehot'
                elif n_unique <= max_categories:
                    encoding_method = 'label'
                else:
                    encoding_method = 'frequency'
            else:
                encoding_method = method
            
            # Apply encoding
            if encoding_method == 'label':
                df = self._label_encode(df, col)
            elif encoding_method == 'onehot':
                df = self._onehot_encode(df, col)
            elif encoding_method == 'frequency':
                df = self._frequency_encode(df, col)
            
            self.transformation_report['encoded_columns'].append({
                'column': col,
                'method': encoding_method,
                'n_categories': n_unique
            })
        
        return df
    
    def _label_encode(self, df: pd.DataFrame, col: str) -> pd.DataFrame:
        """Label encoding for categorical variables"""
        le = LabelEncoder()
        df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
        self.encoders[col] = le
        
        # Drop original column
        df = df.drop(columns=[col])
        logger.info(f"Label encoded '{col}' ({len(le.classes_)} categories)")
        
        return df
    
    def _onehot_encode(self, df: pd.DataFrame, col: str) -> pd.DataFrame:
        """One-hot encoding for categorical variables"""
        # Use pandas get_dummies for simplicity
        dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
        df = pd.concat([df, dummies], axis=1)
        df = df.drop(columns=[col])
        
        logger.info(f"One-hot encoded '{col}' ({len(dummies.columns)} new columns)")
        
        return df
    
    def _frequency_encode(self, df: pd.DataFrame, col: str) -> pd.DataFrame:
        """Frequency encoding for high-cardinality categorical variables"""
        freq_map = df[col].value_counts(normalize=True).to_dict()
        df[f'{col}_freq'] = df[col].map(freq_map)
        df = df.drop(columns=[col])
        
        logger.info(f"Frequency encoded '{col}'")
        
        return df
    
    def _scale_numerical(self, df: pd.DataFrame) -> pd.DataFrame:
        """Scale numerical features"""
        method = self.config.get('transformation.numerical_scaling.method', 'standard')
        
        if method == 'none':
            return df
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Exclude binary columns from scaling
        binary_cols = [col for col in numeric_cols if df[col].nunique() == 2 
                      and set(df[col].unique()).issubset({0, 1})]
        cols_to_scale = [col for col in numeric_cols if col not in binary_cols]
        
        if not cols_to_scale:
            return df
        
        # Select scaler
        scalers = {
            'standard': StandardScaler(),
            'minmax': MinMaxScaler(),
            'robust': RobustScaler()
        }
        
        scaler = scalers.get(method, StandardScaler())
        
        # Fit and transform
        df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])
        self.scalers['numerical'] = scaler
        
        self.transformation_report['scaled_columns'] = cols_to_scale
        logger.info(f"Scaled {len(cols_to_scale)} numerical columns using {method} scaling")
        
        return df
    
    def inverse_transform_scaling(self, df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
        """Inverse transform scaled columns"""
        if 'numerical' in self.scalers:
            df[cols] = self.scalers['numerical'].inverse_transform(df[cols])
        return df
    
    def get_report(self) -> Dict[str, Any]:
        """Get transformation report"""
        return self.transformation_report