"""
Main data processing pipeline orchestration - FULLY OPTIMIZED WITH PyArrow
"""
import pandas as pd
import os
import json
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime
import pyarrow as pa
import pyarrow.parquet as pq
from pyarrow import compute as pc

from src.utils.logger import get_logger, PipelineLogger
from src.utils.config import ConfigManager
from src.ingestion.file_reader import FileReader
from src.ingestion.data_validator import DataValidator
from src.preprocessing.cleaner import DataCleaner
from src.preprocessing.transformer import DataTransformer
from src.quality.profiler import DataProfiler

logger = get_logger()

class DataPipeline:
    """Production-ready data processing pipeline with PyArrow optimization"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the optimized pipeline with PyArrow backend
        """
        self.config = ConfigManager(config_path) if config_path else ConfigManager()
        
        # Initialize logger
        log_path = self.config.get('paths.logs', 'logs') + '/pipeline.log'
        log_level = self.config.get('logging.level', 'INFO')
        PipelineLogger(log_path, log_level)
        
        # Initialize components
        self.reader = FileReader(self.config)
        self.validator = DataValidator()
        self.cleaner = DataCleaner(self.config)
        self.transformer = DataTransformer(self.config)
        self.profiler = DataProfiler()
        
        # PyArrow optimization settings
        self.arrow_enabled = self._check_pyarrow()
        self.use_arrow_string = self.config.get('pyarrow.use_string_dtype', True)
        
        # Pipeline state
        self.pipeline_report = {
            'start_time': None,
            'end_time': None,
            'input_file': None,
            'stages': {},
            'pyarrow_enabled': self.arrow_enabled
        }
    
    def _check_pyarrow(self) -> bool:
        """Verify PyArrow installation and compatibility"""
        try:
            import pyarrow
            logger.info(f"✅ PyArrow detected: v{pyarrow.__version__} - Arrow acceleration ENABLED")
            return True
        except ImportError:
            logger.warning("⚠ PyArrow not found - install with: pip install pyarrow")
            return False
    
    def run(self, 
            input_file: str, 
            output_name: Optional[str] = None,
            skip_validation: bool = False,
            skip_transformation: bool = False) -> Dict[str, Any]:
        """
        Run the complete optimized data pipeline
        """
        self.pipeline_report['start_time'] = datetime.now()
        self.pipeline_report['input_file'] = input_file
        
        logger.info("="*70)
        logger.info("🚀 PRODUCTION DATA PIPELINE WITH PYARROW ACCELERATION")
        logger.info("="*70)
        logger.info(f"Input file: {input_file}")
        logger.info(f"PyArrow: {'✅ ENABLED' if self.arrow_enabled else '❌ DISABLED'}")
        
        try:
            # Stage 1: Ingestion
            logger.info("\n[Stage 1/6] 🚀 INGESTION")
            df = self._stage_ingestion(input_file)
            self.pipeline_report['stages']['ingestion'] = {
                'status': 'success', 
                'rows': len(df), 
                'columns': len(df.columns)
            }
            
            # Stage 2: Initial Profiling
            logger.info("\n[Stage 2/6] 📊 INITIAL PROFILING")
            initial_profile = self._stage_profiling(df, 'initial')
            self.pipeline_report['stages']['initial_profiling'] = {'status': 'success'}
            
            # Stage 3: Validation
            if not skip_validation:
                logger.info("\n[Stage 3/6] ✅ VALIDATION")
                validation_results = self._stage_validation(df)
                self.pipeline_report['stages']['validation'] = validation_results
            else:
                logger.info("\n[Stage 3/6] VALIDATION - SKIPPED")
            
            # Stage 4: Cleaning
            logger.info("\n[Stage 4/6] 🧹 CLEANING")
            df = self._stage_cleaning(df)
            self.pipeline_report['stages']['cleaning'] = {
                'status': 'success', 
                'rows': len(df), 
                'columns': len(df.columns)
            }
            
            # Stage 5: Transformation
            if not skip_transformation:
                logger.info("\n[Stage 5/6] 🔄 TRANSFORMATION")
                df = self._stage_transformation(df)
                self.pipeline_report['stages']['transformation'] = {
                    'status': 'success', 
                    'rows': len(df), 
                    'columns': len(df.columns)
                }
            else:
                logger.info("\n[Stage 5/6] TRANSFORMATION - SKIPPED")
            
            # Stage 6: Final Profiling & Optimized Export
            logger.info("\n[Stage 6/6] 💾 FINAL PROFILING & PYARROW EXPORT")
            final_profile = self._stage_profiling(df, 'final')
            output_paths = self._stage_export_optimized(df, input_file, output_name)
            self.pipeline_report['stages']['export'] = {'status': 'success', 'paths': output_paths}
            
            # Success metrics
            self.pipeline_report['end_time'] = datetime.now()
            duration = (self.pipeline_report['end_time'] - self.pipeline_report['start_time']).total_seconds()
            self.pipeline_report['duration_seconds'] = duration
            self.pipeline_report['status'] = 'success'
            self.pipeline_report['final_shape'] = df.shape
            
            logger.info("="*70)
            logger.info(f"🎉 PIPELINE COMPLETED SUCCESSFULLY in {duration:.2f}s")
            logger.info(f"📈 Final shape: {df.shape}")
            logger.info("="*70)
            
            return self.pipeline_report
            
        except Exception as e:
            logger.error(f"💥 Pipeline failed: {e}")
            self.pipeline_report['status'] = 'failed'
            self.pipeline_report['error'] = str(e)
            raise
    
    def _stage_ingestion(self, input_file: str) -> pd.DataFrame:
        """Stage 1: Optimized file reading"""
        file_info = self.reader.get_file_info(input_file)
        logger.info(f"📁 File: {file_info['name']} ({file_info['size_mb']:.2f} MB)")
        
        df = self.reader.read_file(input_file)
        
        # Apply PyArrow dtype optimization on ingestion
        if self.arrow_enabled and self.use_arrow_string:
            logger.info("⚡ Applying PyArrow string dtype optimization...")
            for col in df.select_dtypes(include=['object']).columns:
                df[col] = df[col].astype("string")
        
        logger.info(f"✅ Loaded {len(df):,} rows × {len(df.columns)} columns")
        return df
    
    def _stage_export_optimized(self, df: pd.DataFrame, input_file: str, output_name: Optional[str] = None) -> Dict[str, str]:
        """Stage 6: Optimized export with PyArrow compression"""
        output_paths = {}
        
        # Determine output name
        if output_name is None:
            output_name = Path(input_file).stem + '_processed'
        
        # Create output directories
        data_dir = Path(self.config.get('paths.processed_data', 'data/processed'))
        report_dir = Path(self.config.get('paths.reports', 'data/reports'))
        data_dir.mkdir(parents=True, exist_ok=True)
        report_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Primary: PyArrow Parquet (fastest, smallest)
        parquet_path = data_dir / f"{output_name}.parquet"
        try:
            # Convert to Arrow Table for maximum efficiency
            table = pa.Table.from_pandas(df, preserve_index=False)
            
            # Write with advanced compression
            pq.write_table(
                table, 
                parquet_path,
                compression='snappy',  # Fast compression
                write_statistics=True,  # Column stats for query optimization
                use_dictionary=True     # Dictionary encoding
            )
            output_paths['parquet'] = str(parquet_path)
            parquet_size = parquet_path.stat().st_size / 1024**2
            logger.info(f"✅ Exported Parquet: {parquet_path} ({parquet_size:.2f} MB)")
            
        except Exception as e:
            logger.warning(f"Parquet export failed: {e}")
            output_paths['parquet'] = None
        
        # 2. CSV Backup (human readable)
        csv_path = data_dir / f"{output_name}.csv"
        csv_size_before = data_dir.stat().st_size / 1024**2 if data_dir.exists() else 0
        df.to_csv(csv_path, index=False)
        csv_size = csv_path.stat().st_size / 1024**2
        output_paths['csv'] = str(csv_path)
        logger.info(f"📄 Exported CSV: {csv_path} ({csv_size:.2f} MB)")
        
        # 3. Metadata & Reports
        if self.config.get('export.include_metadata', True):
            metadata_path = report_dir / f"{output_name}_metadata.json"
            self._save_metadata(df, metadata_path)
            output_paths['metadata'] = str(metadata_path)
        
        if self.config.get('export.include_report', True):
            report_path = report_dir / f"{output_name}_pipeline_report.json"
            self._save_pipeline_report(report_path)
            output_paths['report'] = str(report_path)
        
        logger.info(f"📊 Space savings: Parquet is ~{csv_size/parquet_size:.1f}x smaller than CSV")
        return output_paths
    
    def _stage_profiling(self, df: pd.DataFrame, stage: str) -> Dict[str, Any]:
        """Generate data profile with memory optimization"""
        profile = self.profiler.generate_profile(df)
        
        logger.info(f"✅ Profile generated ({stage}):")
        logger.info(f"  📊 Completeness: {profile['quality_metrics']['completeness']:.2f}%")
        logger.info(f"  ❌ Missing values: {profile['quality_metrics']['total_missing_values']:,}")
        logger.info(f"  💾 Memory: {profile['overview']['memory_usage_mb']:.2f} MB")
        
        profile_path = self._get_output_path('profile', stage)
        self._save_profile(profile, profile_path)
        
        return profile
    
    def _stage_validation(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Stage 3: Validate data"""
        results = self.validator.validate(df)
        
        if results['is_valid']:
            logger.info("✅ Validation passed")
        else:
            logger.warning(f"⚠️  Validation issues: {len(results['errors'])} errors")
        
        validation_path = self._get_output_path('validation', 'report')
        self._save_validation_report(results, validation_path)
        
        return {'status': 'completed', 'is_valid': results['is_valid']}
    
    def _stage_cleaning(self, df: pd.DataFrame) -> pd.DataFrame:
        """Stage 4: Clean data"""
        df_clean = self.cleaner.clean(df)
        report = self.cleaner.get_report()
        logger.info(f"✅ Cleaning completed: {len(report['actions'])} actions performed")
        return df_clean
    
    def _stage_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        """Stage 5: Transform data"""
        df_transformed = self.transformer.transform(df)
        report = self.transformer.get_report()
        logger.info(f"✅ Transformation completed:")
        logger.info(f"  🔢 Encoded columns: {len(report['encoded_columns'])}")
        logger.info(f"  📏 Scaled columns: {len(report['scaled_columns'])}")
        return df_transformed
    
    # Utility methods (unchanged)
    def _get_output_path(self, type: str, name: str, ext: str = 'json') -> Path:
        if type == 'data':
            base_path = Path(self.config.get('paths.processed_data', 'data/processed'))
        elif type in ['profile', 'validation', 'report', 'metadata']:
            base_path = Path(self.config.get('paths.reports', 'data/reports'))
        else:
            base_path = Path('data/output')
        
        base_path.mkdir(parents=True, exist_ok=True)
        return base_path / f"{name}.{ext}"
    
    def _save_profile(self, profile: Dict, path: Path):
        with open(path, 'w') as f:
            json.dump(profile, f, indent=2, default=str)
    
    def _save_validation_report(self, results: Dict, path: Path):
        with open(path, 'w') as f:
            json.dump(results, f, indent=2)
    
    def _save_metadata(self, df: pd.DataFrame, path: Path):
        metadata = {
            'shape': df.shape,
            'columns': df.columns.tolist(),
            'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2,
            'generated_at': datetime.now().isoformat()
        }
        with open(path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def _save_pipeline_report(self, path: Path):
        with open(path, 'w') as f:
            json.dump(self.pipeline_report, f, indent=2, default=str)
