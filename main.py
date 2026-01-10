"""
Main entry point for the data ingestion pipeline
"""
import argparse
from pathlib import Path
from src.pipeline import DataPipeline
from src.utils.logger import get_logger

logger = get_logger()


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description='Data Ingestion and Processing Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process a CSV file
  python main.py data/raw/sales.csv
  
  # Process with custom output name
  python main.py data/raw/sales.csv --output sales_clean
  
  # Process with custom config
  python main.py data/raw/sales.csv --config config/custom_config.yaml
  
  # Skip transformation (cleaning only)
  python main.py data/raw/sales.csv --skip-transformation
        """
    )
    
    parser.add_argument(
        'input_file',
        type=str,
        help='Path to input data file (CSV, Excel, JSON, Parquet)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output file name (default: input_filename_processed)'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='config/pipeline_config.yaml',
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--skip-validation',
        action='store_true',
        help='Skip data validation step'
    )
    
    parser.add_argument(
        '--skip-transformation',
        action='store_true',
        help='Skip data transformation step (encoding, scaling)'
    )
    
    args = parser.parse_args()
    
    # Validate input file exists
    if not Path(args.input_file).exists():
        logger.error(f"Input file not found: {args.input_file}")
        return 1
    
    try:
        # Initialize and run pipeline
        pipeline = DataPipeline(config_path=args.config)
        
        report = pipeline.run(
            input_file=args.input_file,
            output_name=args.output,
            skip_validation=args.skip_validation,
            skip_transformation=args.skip_transformation
        )
        
        # Print summary
        print("\n" + "="*60)
        print("PIPELINE EXECUTION SUMMARY")
        print("="*60)
        print(f"Status: {report['status'].upper()}")
        print(f"Duration: {report['duration_seconds']:.2f} seconds")
        print(f"\nInput: {args.input_file}")
        print(f"Output files:")
        for fmt, path in report['stages']['export']['paths'].items():
            print(f"  - {fmt}: {path}")
        print("="*60)
        
        return 0
    
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())