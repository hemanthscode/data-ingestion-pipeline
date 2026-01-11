"""
Configuration management utility (ENCODING FIXED)
"""
import yaml
from pathlib import Path
import io
from typing import Any, Dict
from src.utils.logger import get_logger

logger = get_logger()

class ConfigManager:
    def __init__(self, config_path: str = "config/pipeline_config.yaml"):
        self.config_path = Path(config_path)
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file with encoding protection"""
        try:
            if not self.config_path.exists():
                logger.warning(f"Config file not found: {self.config_path}. Using ETL defaults.")
                return self._get_etl_default_config()  # ← FIXED DEFAULTS
            
            # FIXED: UTF-8 with error handling
            with open(self.config_path, 'r', encoding='utf-8', errors='ignore') as f:
                config = yaml.safe_load(f)
            
            # FORCE ETL-SAFE DEFAULTS
            config = self._apply_etl_safe_overrides(config)
            
            logger.info(f"✅ Config loaded: {self.config_path}")
            return config
            
        except Exception as e:
            logger.error(f"❌ Config error: {e}. Using ETL defaults.")
            return self._get_etl_default_config()
    
    def _get_etl_default_config(self) -> Dict[str, Any]:
        """ETL-SAFE DEFAULTS - NO SCALING BY DEFAULT"""
        return {
            'paths': {
                'raw_data': 'data/raw',
                'processed_data': 'data/processed',
                'reports': 'data/reports',
                'logs': 'logs'
            },
            'ingestion': {
                'supported_formats': ['csv', 'xlsx', 'xls', 'json', 'parquet'],
                'encoding': 'utf-8',
                'max_file_size_mb': 500,
                'chunk_size': 10000
            },
            'cleaning': {
                'missing_values': {'strategy': 'fill_median', 'threshold': 0.5},
                'duplicates': {'keep': 'first'},
                'outliers': {'method': 'iqr', 'threshold': 1.5, 'action': 'cap'}
            },
            'transformation': {
                'categorical_encoding': {'method': 'onehot', 'max_categories': 50},
                'numerical_scaling': {
                    'method': 'none',  # ← FIXED: ETL SAFE
                    'business_columns': ['age', 'salary', 'phone', 'employee_id']
                },
                'feature_engineering': {
                    'create_datetime_features': True
                }
            },
            'export': {
                'formats': ['csv', 'parquet'],
                'include_metadata': True,
                'include_report': True,
                'compression': 'gzip'
            }
        }
    
    def _apply_etl_safe_overrides(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Force ETL-safe settings"""
        # FORCE NO SCALING for ETL pipeline
        if config.get('transformation', {}).get('numerical_scaling', {}).get('method') not in ['none']:
            config['transformation']['numerical_scaling'] = {
                'method': 'none',
                'business_columns': ['age', 'salary', 'phone', 'employee_id']
            }
            logger.warning("🔒 Forced ETL-safe: numerical_scaling = 'none'")
        
        return config
    
    def get(self, key_path: str, default: Any = None) -> Any:
        keys = key_path.split('.')
        value = self.config
        for key in keys:
            if isinstance(value, dict):
                value = value.get(key)
                if value is None:
                    return default
            else:
                return default
        return value

    def update(self, key_path: str, value: Any):
        keys = key_path.split('.')
        config = self.config
        for key in keys[:-1]:
            config = config.setdefault(key, {})
        config[keys[-1]] = value
        logger.info(f"Updated config: {key_path} = {value}")
    
    def save(self, path: str = None):
        save_path = Path(path) if path else self.config_path
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True)
        logger.info(f"Configuration saved to {save_path}")
