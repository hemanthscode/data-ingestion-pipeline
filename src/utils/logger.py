"""
Logging utility for the data pipeline
"""
import sys
from pathlib import Path
from loguru import logger
from typing import Optional


class PipelineLogger:
    """Centralized logging configuration"""
    
    def __init__(self, log_path: Optional[str] = None, level: str = "INFO"):
        """
        Initialize logger
        
        Args:
            log_path: Path to log file
            level: Logging level (DEBUG, INFO, WARNING, ERROR)
        """
        self.level = level
        self.log_path = log_path or "logs/pipeline.log"
        self._setup_logger()
    
    def _setup_logger(self):
        """Configure logger with file and console handlers"""
        # Remove default handler
        logger.remove()
        
        # Add console handler with color
        logger.add(
            sys.stdout,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
            level=self.level,
            colorize=True
        )
        
        # Add file handler with rotation
        Path(self.log_path).parent.mkdir(parents=True, exist_ok=True)
        logger.add(
            self.log_path,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            level=self.level,
            rotation="10 MB",
            retention="30 days",
            compression="zip"
        )
    
    @staticmethod
    def get_logger():
        """Get logger instance"""
        return logger


# Global logger instance
def get_logger():
    """Get the global logger instance"""
    return logger