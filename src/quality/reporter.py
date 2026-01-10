"""
Quality reporting and visualization module
"""
import pandas as pd
import json
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
from src.utils.logger import get_logger
from src.utils.config import ConfigManager

logger = get_logger()


class QualityReporter:
    """Generate comprehensive quality reports"""
    
    def __init__(self, config: Optional[ConfigManager] = None):
        """
        Initialize quality reporter
        
        Args:
            config: Configuration manager instance
        """
        self.config = config or ConfigManager()
        self.report_data = {}
    
    def generate_report(self, 
                       df_before: pd.DataFrame,
                       df_after: pd.DataFrame,
                       validation_results: Optional[Dict] = None,
                       cleaning_report: Optional[Dict] = None,
                       transformation_report: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Generate comprehensive quality report
        
        Args:
            df_before: DataFrame before processing
            df_after: DataFrame after processing
            validation_results: Validation results
            cleaning_report: Cleaning report
            transformation_report: Transformation report
        
        Returns:
            Complete quality report
        """
        logger.info("Generating quality report...")
        
        self.report_data = {
            'metadata': self._get_metadata(),
            'comparison': self._compare_datasets(df_before, df_after),
            'validation': validation_results or {},
            'cleaning': cleaning_report or {},
            'transformation': transformation_report or {},
            'quality_metrics': self._calculate_quality_metrics(df_before, df_after),
            'recommendations': self._generate_recommendations(df_after)
        }
        
        logger.info("Quality report generated")
        return self.report_data
    
    def _get_metadata(self) -> Dict[str, Any]:
        """Get report metadata"""
        return {
            'generated_at': datetime.now().isoformat(),
            'pipeline_version': '1.0.0',
            'python_version': '3.10+'
        }
    
    def _compare_datasets(self, df_before: pd.DataFrame, df_after: pd.DataFrame) -> Dict[str, Any]:
        """Compare before and after datasets"""
        return {
            'before': {
                'rows': len(df_before),
                'columns': len(df_before.columns),
                'memory_mb': round(df_before.memory_usage(deep=True).sum() / (1024**2), 2),
                'missing_values': int(df_before.isna().sum().sum()),
                'duplicates': int(df_before.duplicated().sum())
            },
            'after': {
                'rows': len(df_after),
                'columns': len(df_after.columns),
                'memory_mb': round(df_after.memory_usage(deep=True).sum() / (1024**2), 2),
                'missing_values': int(df_after.isna().sum().sum()),
                'duplicates': int(df_after.duplicated().sum())
            },
            'changes': {
                'rows_removed': len(df_before) - len(df_after),
                'rows_removed_pct': round(((len(df_before) - len(df_after)) / len(df_before)) * 100, 2),
                'columns_added': len(df_after.columns) - len(df_before.columns),
                'missing_values_reduced': int(df_before.isna().sum().sum() - df_after.isna().sum().sum()),
                'duplicates_removed': int(df_before.duplicated().sum() - df_after.duplicated().sum()),
                'memory_saved_mb': round((df_before.memory_usage(deep=True).sum() - 
                                        df_after.memory_usage(deep=True).sum()) / (1024**2), 2)
            }
        }
    
    def _calculate_quality_metrics(self, df_before: pd.DataFrame, df_after: pd.DataFrame) -> Dict[str, Any]:
        """Calculate quality improvement metrics"""
        total_cells_before = df_before.shape[0] * df_before.shape[1]
        total_cells_after = df_after.shape[0] * df_after.shape[1]
        
        completeness_before = ((total_cells_before - df_before.isna().sum().sum()) / total_cells_before) * 100
        completeness_after = ((total_cells_after - df_after.isna().sum().sum()) / total_cells_after) * 100
        
        return {
            'completeness_before': round(completeness_before, 2),
            'completeness_after': round(completeness_after, 2),
            'completeness_improvement': round(completeness_after - completeness_before, 2),
            'duplicate_ratio_before': round((df_before.duplicated().sum() / len(df_before)) * 100, 2),
            'duplicate_ratio_after': round((df_after.duplicated().sum() / len(df_after)) * 100, 2),
            'data_quality_score': self._calculate_quality_score(df_after)
        }
    
    def _calculate_quality_score(self, df: pd.DataFrame) -> float:
        """
        Calculate overall data quality score (0-100)
        
        Factors:
        - Completeness (40%)
        - Uniqueness (30%)
        - Consistency (20%)
        - Validity (10%)
        """
        # Completeness score
        total_cells = df.shape[0] * df.shape[1]
        completeness = ((total_cells - df.isna().sum().sum()) / total_cells) * 100 if total_cells > 0 else 0
        
        # Uniqueness score
        uniqueness = (1 - (df.duplicated().sum() / len(df))) * 100 if len(df) > 0 else 0
        
        # Consistency score (data type consistency)
        consistency = 100  # Assume consistent after processing
        
        # Validity score (assume valid after processing)
        validity = 100
        
        # Weighted average
        quality_score = (
            completeness * 0.4 +
            uniqueness * 0.3 +
            consistency * 0.2 +
            validity * 0.1
        )
        
        return round(quality_score, 2)
    
    def _generate_recommendations(self, df: pd.DataFrame) -> list:
        """Generate recommendations based on data analysis"""
        recommendations = []
        
        # Check for high cardinality columns
        for col in df.select_dtypes(include=['object']).columns:
            unique_ratio = df[col].nunique() / len(df)
            if unique_ratio > 0.9:
                recommendations.append({
                    'type': 'warning',
                    'column': col,
                    'message': f"High cardinality detected ({df[col].nunique()} unique values). Consider if this should be an ID column."
                })
        
        # Check for remaining missing values
        missing_cols = df.columns[df.isna().any()].tolist()
        if missing_cols:
            recommendations.append({
                'type': 'info',
                'columns': missing_cols,
                'message': f"{len(missing_cols)} columns still have missing values. Consider additional imputation strategies."
            })
        
        # Check for potential date columns stored as strings
        for col in df.select_dtypes(include=['object']).columns:
            if df[col].astype(str).str.match(r'\d{4}-\d{2}-\d{2}').sum() > len(df) * 0.5:
                recommendations.append({
                    'type': 'suggestion',
                    'column': col,
                    'message': f"Column appears to contain dates but is stored as string. Consider converting to datetime."
                })
        
        # Check for imbalanced data
        for col in df.select_dtypes(include=['int64', 'float64']).columns:
            if df[col].nunique() == 2:  # Binary column
                value_counts = df[col].value_counts(normalize=True)
                if value_counts.min() < 0.1:
                    recommendations.append({
                        'type': 'warning',
                        'column': col,
                        'message': f"Highly imbalanced binary column ({value_counts.min()*100:.1f}% minority class). Consider resampling techniques."
                    })
        
        # Check memory usage
        memory_mb = df.memory_usage(deep=True).sum() / (1024**2)
        if memory_mb > 100:
            recommendations.append({
                'type': 'optimization',
                'message': f"High memory usage ({memory_mb:.1f} MB). Consider using categorical dtypes or downcasting numerical types."
            })
        
        return recommendations
    
    def save_report(self, output_name: str, format: str = 'json') -> str:
        """
        Save report to file
        
        Args:
            output_name: Output file name
            format: Output format ('json', 'html', 'txt')
        
        Returns:
            Path to saved report
        """
        reports_dir = Path(self.config.get('paths.reports', 'data/reports'))
        reports_dir.mkdir(parents=True, exist_ok=True)
        
        if format == 'json':
            file_path = reports_dir / f"{output_name}_quality_report.json"
            with open(file_path, 'w') as f:
                json.dump(self.report_data, f, indent=2, default=str)
        
        elif format == 'html':
            file_path = reports_dir / f"{output_name}_quality_report.html"
            html_content = self._generate_html_report()
            with open(file_path, 'w') as f:
                f.write(html_content)
        
        elif format == 'txt':
            file_path = reports_dir / f"{output_name}_quality_report.txt"
            txt_content = self._generate_text_report()
            with open(file_path, 'w') as f:
                f.write(txt_content)
        
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Quality report saved to {file_path}")
        return str(file_path)
    
    def _generate_html_report(self) -> str:
        """Generate HTML report"""
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Data Quality Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background-color: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        h1 {{ color: #333; border-bottom: 3px solid #4CAF50; padding-bottom: 10px; }}
        h2 {{ color: #555; margin-top: 30px; }}
        .metric {{ background-color: #f9f9f9; padding: 15px; margin: 10px 0; border-left: 4px solid #4CAF50; }}
        .metric-label {{ font-weight: bold; color: #666; }}
        .metric-value {{ font-size: 24px; color: #333; }}
        .improvement {{ color: #4CAF50; }}
        .warning {{ color: #ff9800; }}
        .error {{ color: #f44336; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th {{ background-color: #4CAF50; color: white; padding: 12px; text-align: left; }}
        td {{ padding: 10px; border-bottom: 1px solid #ddd; }}
        tr:hover {{ background-color: #f5f5f5; }}
        .recommendation {{ padding: 10px; margin: 10px 0; border-radius: 4px; }}
        .rec-warning {{ background-color: #fff3cd; border-left: 4px solid #ff9800; }}
        .rec-info {{ background-color: #d1ecf1; border-left: 4px solid #17a2b8; }}
        .rec-suggestion {{ background-color: #d4edda; border-left: 4px solid #28a745; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 Data Quality Report</h1>
        <p><strong>Generated:</strong> {self.report_data['metadata']['generated_at']}</p>
        
        <h2>Dataset Comparison</h2>
        <table>
            <tr>
                <th>Metric</th>
                <th>Before</th>
                <th>After</th>
                <th>Change</th>
            </tr>
            <tr>
                <td>Rows</td>
                <td>{self.report_data['comparison']['before']['rows']:,}</td>
                <td>{self.report_data['comparison']['after']['rows']:,}</td>
                <td class="{'improvement' if self.report_data['comparison']['changes']['rows_removed'] >= 0 else 'warning'}">
                    {self.report_data['comparison']['changes']['rows_removed']:+,}
                </td>
            </tr>
            <tr>
                <td>Columns</td>
                <td>{self.report_data['comparison']['before']['columns']}</td>
                <td>{self.report_data['comparison']['after']['columns']}</td>
                <td>{self.report_data['comparison']['changes']['columns_added']:+}</td>
            </tr>
            <tr>
                <td>Missing Values</td>
                <td>{self.report_data['comparison']['before']['missing_values']:,}</td>
                <td>{self.report_data['comparison']['after']['missing_values']:,}</td>
                <td class="improvement">{-self.report_data['comparison']['changes']['missing_values_reduced']:,}</td>
            </tr>
            <tr>
                <td>Duplicates</td>
                <td>{self.report_data['comparison']['before']['duplicates']:,}</td>
                <td>{self.report_data['comparison']['after']['duplicates']:,}</td>
                <td class="improvement">{-self.report_data['comparison']['changes']['duplicates_removed']:,}</td>
            </tr>
        </table>
        
        <h2>Quality Metrics</h2>
        <div class="metric">
            <div class="metric-label">Data Quality Score</div>
            <div class="metric-value improvement">{self.report_data['quality_metrics']['data_quality_score']}/100</div>
        </div>
        <div class="metric">
            <div class="metric-label">Completeness</div>
            <div class="metric-value">{self.report_data['quality_metrics']['completeness_after']}%</div>
        </div>
        
        <h2>Recommendations</h2>
        {''.join([f'<div class="recommendation rec-{rec["type"]}">{rec["message"]}</div>' 
                  for rec in self.report_data['recommendations']])}
    </div>
</body>
</html>
"""
        return html
    
    def _generate_text_report(self) -> str:
        """Generate plain text report"""
        lines = []
        lines.append("=" * 80)
        lines.append("DATA QUALITY REPORT")
        lines.append("=" * 80)
        lines.append(f"Generated: {self.report_data['metadata']['generated_at']}")
        lines.append("")
        
        lines.append("DATASET COMPARISON")
        lines.append("-" * 80)
        comp = self.report_data['comparison']
        lines.append(f"Rows:           {comp['before']['rows']:>10,} → {comp['after']['rows']:>10,} ({comp['changes']['rows_removed']:+,})")
        lines.append(f"Columns:        {comp['before']['columns']:>10} → {comp['after']['columns']:>10} ({comp['changes']['columns_added']:+})")
        lines.append(f"Missing Values: {comp['before']['missing_values']:>10,} → {comp['after']['missing_values']:>10,} ({-comp['changes']['missing_values_reduced']:+,})")
        lines.append(f"Duplicates:     {comp['before']['duplicates']:>10,} → {comp['after']['duplicates']:>10,} ({-comp['changes']['duplicates_removed']:+,})")
        lines.append("")
        
        lines.append("QUALITY METRICS")
        lines.append("-" * 80)
        qm = self.report_data['quality_metrics']
        lines.append(f"Data Quality Score: {qm['data_quality_score']}/100")
        lines.append(f"Completeness:       {qm['completeness_after']}% (improvement: {qm['completeness_improvement']:+.2f}%)")
        lines.append("")
        
        if self.report_data['recommendations']:
            lines.append("RECOMMENDATIONS")
            lines.append("-" * 80)
            for i, rec in enumerate(self.report_data['recommendations'], 1):
                lines.append(f"{i}. [{rec['type'].upper()}] {rec['message']}")
        
        lines.append("=" * 80)
        
        return "\n".join(lines)
    
    def print_summary(self):
        """Print report summary to console"""
        print(self._generate_text_report())