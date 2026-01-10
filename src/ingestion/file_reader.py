"""
File reading and ingestion module
"""
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Any
from src.utils.logger import get_logger
from src.utils.config import ConfigManager

logger = get_logger()


class FileReader:
    """Read various file formats into pandas DataFrame"""
    
    def __init__(self, config: Optional[ConfigManager] = None):
        """
        Initialize file reader
        
        Args:
            config: Configuration manager instance
        """
        self.config = config or ConfigManager()
        self.supported_formats = self.config.get('ingestion.supported_formats', 
                                                  ['csv', 'xlsx', 'xls', 'json', 'parquet'])
    
    def read_file(self, file_path: str, **kwargs) -> pd.DataFrame:
        """
        Read file based on extension
        
        Args:
            file_path: Path to the file
            **kwargs: Additional arguments for pandas readers
        
        Returns:
            DataFrame containing the data
        
        Raises:
            ValueError: If file format not supported
            FileNotFoundError: If file doesn't exist
        """
        path = Path(file_path)
        
        if not path.exists():
            logger.error(f"File not found: {file_path}")
            raise FileNotFoundError(f"File not found: {file_path}")
        
        file_ext = path.suffix.lower().lstrip('.')
        
        if file_ext not in self.supported_formats:
            logger.error(f"Unsupported file format: {file_ext}")
            raise ValueError(f"Unsupported format: {file_ext}. Supported: {self.supported_formats}")
        
        # Check file size
        file_size_mb = path.stat().st_size / (1024 * 1024)
        max_size = self.config.get('ingestion.max_file_size_mb', 500)
        
        if file_size_mb > max_size:
            logger.warning(f"Large file detected: {file_size_mb:.2f} MB")
        
        logger.info(f"Reading file: {file_path} (Format: {file_ext}, Size: {file_size_mb:.2f} MB)")
        
        try:
            df = self._read_by_format(file_path, file_ext, **kwargs)
            logger.info(f"Successfully read {len(df)} rows and {len(df.columns)} columns")
            return df
        
        except Exception as e:
            logger.error(f"Error reading file: {e}")
            raise
    
    def _read_by_format(self, file_path: str, file_ext: str, **kwargs) -> pd.DataFrame:
        """
        Read file using appropriate pandas method
        
        Args:
            file_path: Path to file
            file_ext: File extension
            **kwargs: Additional arguments
        
        Returns:
            DataFrame
        """
        encoding = self.config.get('ingestion.encoding', 'utf-8')
        chunk_size = self.config.get('ingestion.chunk_size')
        
        readers = {
            'csv': lambda: pd.read_csv(file_path, encoding=encoding, **kwargs),
            'xlsx': lambda: pd.read_excel(file_path, engine='openpyxl', **kwargs),
            'xls': lambda: pd.read_excel(file_path, engine='xlrd', **kwargs),
            'json': lambda: pd.read_json(file_path, **kwargs),
            'parquet': lambda: pd.read_parquet(file_path, **kwargs)
        }
        
        return readers[file_ext]()
    
    def detect_delimiter(self, file_path: str, sample_rows: int = 5) -> str:
        """
        Auto-detect CSV delimiter
        
        Args:
            file_path: Path to CSV file
            sample_rows: Number of rows to sample
        
        Returns:
            Detected delimiter
        """
        import csv
        
        with open(file_path, 'r', encoding='utf-8') as f:
            sample = ''.join([f.readline() for _ in range(sample_rows)])
            sniffer = csv.Sniffer()
            delimiter = sniffer.sniff(sample).delimiter
        
        logger.info(f"Detected delimiter: '{delimiter}'")
        return delimiter
    
    def get_file_info(self, file_path: str) -> Dict[str, Any]:
        """
        Get file metadata
        
        Args:
            file_path: Path to file
        
        Returns:
            Dictionary with file information
        """
        path = Path(file_path)
        
        return {
            'name': path.name,
            'extension': path.suffix.lstrip('.'),
            'size_mb': path.stat().st_size / (1024 * 1024),
            'modified': path.stat().st_mtime,
            'absolute_path': str(path.absolute())
        }