"""
Data export module for various formats
"""
import pandas as pd
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
from src.utils.logger import get_logger
from src.utils.config import ConfigManager

logger = get_logger()


class DataExporter:
    """Export processed data to various formats"""
    
    def __init__(self, config: Optional[ConfigManager] = None):
        """
        Initialize data exporter
        
        Args:
            config: Configuration manager instance
        """
        self.config = config or ConfigManager()
        self.export_paths = {}
    
    def export(self, 
               df: pd.DataFrame, 
               output_name: str,
               formats: Optional[List[str]] = None,
               metadata: Optional[Dict[str, Any]] = None,
               reports: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
        """
        Export data to multiple formats
        
        Args:
            df: DataFrame to export
            output_name: Base name for output files
            formats: List of formats to export ('csv', 'parquet', 'excel', 'json')
            metadata: Optional metadata to export
            reports: Optional reports to export
        
        Returns:
            Dictionary mapping format to file path
        """
        logger.info(f"Exporting data as '{output_name}'...")
        
        # Get export formats from config if not specified
        if formats is None:
            formats = self.config.get('export.formats', ['csv', 'parquet'])
        
        # Create output directory
        output_dir = Path(self.config.get('paths.processed_data', 'data/processed'))
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Export data in specified formats
        for fmt in formats:
            try:
                if fmt == 'csv':
                    self._export_csv(df, output_dir, output_name)
                elif fmt == 'parquet':
                    self._export_parquet(df, output_dir, output_name)
                elif fmt == 'excel':
                    self._export_excel(df, output_dir, output_name)
                elif fmt == 'json':
                    self._export_json(df, output_dir, output_name)
                elif fmt == 'feather':
                    self._export_feather(df, output_dir, output_name)
                elif fmt == 'hdf':
                    self._export_hdf(df, output_dir, output_name)
                else:
                    logger.warning(f"Unknown export format: {fmt}")
            except Exception as e:
                logger.error(f"Failed to export as {fmt}: {e}")
        
        # Export metadata if provided
        if metadata and self.config.get('export.include_metadata', True):
            self._export_metadata(metadata, output_dir, output_name)
        
        # Export reports if provided
        if reports and self.config.get('export.include_report', True):
            self._export_reports(reports, output_dir, output_name)
        
        logger.info(f"Export completed. {len(self.export_paths)} files created.")
        return self.export_paths
    
    def _export_csv(self, df: pd.DataFrame, output_dir: Path, output_name: str):
        """Export to CSV format"""
        compression = self.config.get('export.compression', 'gzip')
        
        if compression and compression != 'none':
            file_path = output_dir / f"{output_name}.csv.gz"
            df.to_csv(file_path, index=False, compression=compression)
        else:
            file_path = output_dir / f"{output_name}.csv"
            df.to_csv(file_path, index=False)
        
        self.export_paths['csv'] = str(file_path)
        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        logger.info(f"✓ CSV exported: {file_path} ({file_size_mb:.2f} MB)")
    
    def _export_parquet(self, df: pd.DataFrame, output_dir: Path, output_name: str):
        """Export to Parquet format"""
        file_path = output_dir / f"{output_name}.parquet"
        
        # Parquet has built-in compression
        df.to_parquet(
            file_path, 
            index=False,
            engine='pyarrow',
            compression='snappy'
        )
        
        self.export_paths['parquet'] = str(file_path)
        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        logger.info(f"✓ Parquet exported: {file_path} ({file_size_mb:.2f} MB)")
    
    def _export_excel(self, df: pd.DataFrame, output_dir: Path, output_name: str):
        """Export to Excel format"""
        file_path = output_dir / f"{output_name}.xlsx"
        
        with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Data', index=False)
            
            # Auto-adjust column widths
            worksheet = writer.sheets['Data']
            for idx, col in enumerate(df.columns):
                max_length = max(
                    df[col].astype(str).apply(len).max(),
                    len(str(col))
                )
                worksheet.column_dimensions[chr(65 + idx)].width = min(max_length + 2, 50)
        
        self.export_paths['excel'] = str(file_path)
        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        logger.info(f"✓ Excel exported: {file_path} ({file_size_mb:.2f} MB)")
    
    def _export_json(self, df: pd.DataFrame, output_dir: Path, output_name: str):
        """Export to JSON format"""
        file_path = output_dir / f"{output_name}.json"
        
        # Export as records (list of dictionaries)
        df.to_json(
            file_path, 
            orient='records', 
            indent=2,
            date_format='iso'
        )
        
        self.export_paths['json'] = str(file_path)
        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        logger.info(f"✓ JSON exported: {file_path} ({file_size_mb:.2f} MB)")
    
    def _export_feather(self, df: pd.DataFrame, output_dir: Path, output_name: str):
        """Export to Feather format (fast I/O)"""
        file_path = output_dir / f"{output_name}.feather"
        df.to_feather(file_path)
        
        self.export_paths['feather'] = str(file_path)
        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        logger.info(f"✓ Feather exported: {file_path} ({file_size_mb:.2f} MB)")
    
    def _export_hdf(self, df: pd.DataFrame, output_dir: Path, output_name: str):
        """Export to HDF5 format"""
        file_path = output_dir / f"{output_name}.h5"
        df.to_hdf(file_path, key='data', mode='w', complevel=9)
        
        self.export_paths['hdf'] = str(file_path)
        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        logger.info(f"✓ HDF5 exported: {file_path} ({file_size_mb:.2f} MB)")
    
    def _export_metadata(self, metadata: Dict[str, Any], output_dir: Path, output_name: str):
        """Export metadata as JSON"""
        reports_dir = Path(self.config.get('paths.reports', 'data/reports'))
        reports_dir.mkdir(parents=True, exist_ok=True)
        
        file_path = reports_dir / f"{output_name}_metadata.json"
        
        # Add export timestamp
        metadata['exported_at'] = datetime.now().isoformat()
        
        with open(file_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        self.export_paths['metadata'] = str(file_path)
        logger.info(f"✓ Metadata exported: {file_path}")
    
    def _export_reports(self, reports: Dict[str, Any], output_dir: Path, output_name: str):
        """Export quality reports as JSON"""
        reports_dir = Path(self.config.get('paths.reports', 'data/reports'))
        reports_dir.mkdir(parents=True, exist_ok=True)
        
        file_path = reports_dir / f"{output_name}_report.json"
        
        with open(file_path, 'w') as f:
            json.dump(reports, f, indent=2, default=str)
        
        self.export_paths['report'] = str(file_path)
        logger.info(f"✓ Report exported: {file_path}")
    
    def export_data_dictionary(self, 
                               df: pd.DataFrame, 
                               output_name: str,
                               descriptions: Optional[Dict[str, str]] = None) -> str:
        """
        Export data dictionary (schema documentation)
        
        Args:
            df: DataFrame to document
            output_name: Output file name
            descriptions: Optional column descriptions
        
        Returns:
            Path to exported dictionary
        """
        reports_dir = Path(self.config.get('paths.reports', 'data/reports'))
        reports_dir.mkdir(parents=True, exist_ok=True)
        
        file_path = reports_dir / f"{output_name}_dictionary.csv"
        
        # Create data dictionary
        dictionary = []
        for col in df.columns:
            entry = {
                'column_name': col,
                'data_type': str(df[col].dtype),
                'non_null_count': df[col].notna().sum(),
                'null_count': df[col].isna().sum(),
                'unique_values': df[col].nunique(),
                'description': descriptions.get(col, '') if descriptions else ''
            }
            
            # Add sample values for categorical
            if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_categorical_dtype(df[col]):
                sample_values = df[col].value_counts().head(5).index.tolist()
                entry['sample_values'] = ', '.join(map(str, sample_values))
            
            # Add statistics for numerical
            elif pd.api.types.is_numeric_dtype(df[col]):
                entry['min'] = df[col].min()
                entry['max'] = df[col].max()
                entry['mean'] = df[col].mean()
                entry['median'] = df[col].median()
            
            dictionary.append(entry)
        
        # Save as CSV
        dict_df = pd.DataFrame(dictionary)
        dict_df.to_csv(file_path, index=False)
        
        logger.info(f"✓ Data dictionary exported: {file_path}")
        return str(file_path)
    
    def export_sample(self, 
                     df: pd.DataFrame, 
                     output_name: str, 
                     n_rows: int = 100) -> str:
        """
        Export a sample of the data
        
        Args:
            df: DataFrame to sample
            output_name: Output file name
            n_rows: Number of rows to sample
        
        Returns:
            Path to exported sample
        """
        output_dir = Path(self.config.get('paths.processed_data', 'data/processed'))
        output_dir.mkdir(parents=True, exist_ok=True)
        
        file_path = output_dir / f"{output_name}_sample.csv"
        
        sample_df = df.sample(n=min(n_rows, len(df)), random_state=42)
        sample_df.to_csv(file_path, index=False)
        
        logger.info(f"✓ Sample exported: {file_path} ({len(sample_df)} rows)")
        return str(file_path)
    
    def get_export_summary(self) -> Dict[str, Any]:
        """Get summary of exported files"""
        summary = {
            'total_files': len(self.export_paths),
            'files': self.export_paths,
            'total_size_mb': 0
        }
        
        # Calculate total size
        for path in self.export_paths.values():
            if Path(path).exists():
                summary['total_size_mb'] += Path(path).stat().st_size / (1024 * 1024)
        
        summary['total_size_mb'] = round(summary['total_size_mb'], 2)
        
        return summary