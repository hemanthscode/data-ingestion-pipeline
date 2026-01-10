"""
Configuration management utility
"""
import yaml
from pathlib import Path
from typing import Any, Dict
from src.utils.logger import get_logger

logger = get_logger()


class ConfigManager:
    """Manage pipeline configuration"""
    
    def __init__(self, config_path: str = "config/pipeline_config.yaml"):
        """
        Initialize configuration manager
        
        Args:
            config_path: Path to configuration YAML file
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            if not self.config_path.exists():
                logger.warning(f"Config file not found: {self.config_path}. Using defaults.")
                return self._get_default_config()
            
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            logger.info(f"Configuration loaded from {self.config_path}")
            return config
        
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Return default configuration"""
        return {
            'paths': {
                'raw_data': 'data/raw',
                'processed_data': 'data/processed',
                'reports': 'data/reports',
                'logs': 'logs'
            },
            'ingestion': {
                'supported_formats': ['csv', 'xlsx', 'xls', 'json'],
                'encoding': 'utf-8',
                'max_file_size_mb': 500
            },
            'cleaning': {
                'missing_values': {'strategy': 'auto', 'threshold': 0.5},
                'duplicates': {'keep': 'first'},
                'outliers': {'method': 'iqr', 'threshold': 1.5, 'action': 'cap'}
            },
            'transformation': {
                'categorical_encoding': {'method': 'auto'},
                'numerical_scaling': {'method': 'standard'}
            },
            'export': {
                'formats': ['csv', 'parquet'],
                'include_metadata': True,
                'include_report': True
            }
        }
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation
        
        Args:
            key_path: Dot-separated path (e.g., 'cleaning.missing_values.strategy')
            default: Default value if key not found
        
        Returns:
            Configuration value
        """
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
        """
        Update configuration value
        
        Args:
            key_path: Dot-separated path
            value: New value
        """
        keys = key_path.split('.')
        config = self.config
        
        for key in keys[:-1]:
            config = config.setdefault(key, {})
        
        config[keys[-1]] = value
        logger.info(f"Updated config: {key_path} = {value}")
    
    def save(self, path: str = None):
        """Save configuration to file"""
        save_path = Path(path) if path else self.config_path
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
        
        logger.info(f"Configuration saved to {save_path}")