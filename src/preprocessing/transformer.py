"""
Data transformation module (FULLY FIXED - ETL SAFE)
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
    """Transform data for analysis/ML - BUSINESS SAFE"""
    
    def __init__(self, config: Optional[ConfigManager] = None):
        self.config = config or ConfigManager()
        self.encoders = {}
        self.scalers = {}
        self.transformation_report = {
            'encoded_columns': [],
            'scaled_columns': [],
            'skipped_scaling': [],
            'engineered_features': []
        }
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Perform complete transformation pipeline"""
        logger.info("🔄 Starting safe data transformation...")
        df_transformed = df.copy()
        
        # Feature engineering (safe)
        df_transformed = self._engineer_features(df_transformed)
        
        # Encode categorical variables (safe - SKIP PHONE)
        df_transformed = self._encode_categorical(df_transformed)
        
        # Scale numerical features (BUSINESS SAFE)
        df_transformed = self._scale_numerical(df_transformed)
        
        logger.info("✅ Transformation complete - business readable")
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
            logger.info(f"📅 Created datetime features from '{col}'")
        return df
    
    def _encode_categorical(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical variables - SKIP PHONE NUMBERS"""
        method = self.config.get('transformation.categorical_encoding.method', 'auto')
        max_categories = self.config.get('transformation.categorical_encoding.max_categories', 50)
        
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # PROTECT PHONE COLUMNS
        protected_cols = [col for col in categorical_cols if 'phone' in col.lower()]
        for col in protected_cols:
            logger.info(f"🛡️ Skipping encoding for phone column: {col}")
            categorical_cols.remove(col)
        
        for col in categorical_cols:
            n_unique = df[col].nunique()
            
            if method == 'auto':
                if n_unique <= 10:
                    encoding_method = 'onehot'
                elif n_unique <= max_categories:
                    encoding_method = 'label'
                else:
                    encoding_method = 'frequency'
            else:
                encoding_method = method
            
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
        le = LabelEncoder()
        df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
        self.encoders[col] = le
        df = df.drop(columns=[col])
        logger.info(f"🔢 Label encoded '{col}'")
        return df
    
    def _onehot_encode(self, df: pd.DataFrame, col: str) -> pd.DataFrame:
        dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
        df = pd.concat([df, dummies], axis=1)
        df = df.drop(columns=[col])
        logger.info(f"🔢 One-hot encoded '{col}' ({len(dummies.columns)} new columns)")
        return df
    
    def _frequency_encode(self, df: pd.DataFrame, col: str) -> pd.DataFrame:
        freq_map = df[col].value_counts(normalize=True).to_dict()
        df[f'{col}_freq'] = df[col].map(freq_map)
        df = df.drop(columns=[col])
        logger.info(f"📊 Frequency encoded '{col}'")
        return df
    
    def _scale_numerical(self, df: pd.DataFrame) -> pd.DataFrame:
        """Scale ONLY ML columns - PROTECT BUSINESS DATA"""
        method = self.config.get('transformation.numerical_scaling.method', 'none')
        
        if method == 'none':
            logger.info("✅ ETL MODE: Skipping numerical scaling (Age/Salary preserved)")
            return df
        
        # BUSINESS COLUMNS - NEVER SCALE
        business_columns = self.config.get('transformation.numerical_scaling.business_columns', [])
        business_columns.extend(['age', 'salary', 'phone', 'employee_id', 'id'])
        
        # Find ML columns only
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        ml_cols = [col for col in numeric_cols if col.lower() not in 
                   [b.lower() for b in business_columns]]
        
        if not ml_cols:
            logger.info("✅ No ML columns found - keeping business data intact")
            self.transformation_report['skipped_scaling'] = business_columns
            return df
        
        # Scale ML columns only
        from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
        scalers = {
            'standard': StandardScaler(),
            'minmax': MinMaxScaler(),
            'robust': RobustScaler()
        }
        scaler = scalers.get(method, StandardScaler())
        
        df[ml_cols] = scaler.fit_transform(df[ml_cols])
        self.scalers['numerical'] = scaler
        self.transformation_report['scaled_columns'] = ml_cols
        
        logger.info(f"⚖️ Scaled {len(ml_cols)} ML columns: {ml_cols}")
        logger.info(f"🛡️ Protected business columns: {business_columns}")
        return df
    
    def inverse_transform_scaling(self, df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
        if 'numerical' in self.scalers:
            df[cols] = self.scalers['numerical'].inverse_transform(df[cols])
        return df
    
    def get_report(self) -> Dict[str, Any]:
        return self.transformation_report
