"""
Example usage of the Data Pipeline
This script demonstrates various ways to use the pipeline
"""
import pandas as pd
import numpy as np
from pathlib import Path
from src.pipeline import DataPipeline
from src.utils.config import ConfigManager


def create_sample_data():
    """Create sample dataset for demonstration"""
    np.random.seed(42)
    
    n_rows = 1000
    
    data = {
        'customer_id': range(1, n_rows + 1),
        'name': [f'Customer_{i}' for i in range(n_rows)],
        'age': np.random.randint(18, 80, n_rows),
        'income': np.random.normal(50000, 20000, n_rows),
        'purchase_amount': np.random.exponential(100, n_rows),
        'category': np.random.choice(['A', 'B', 'C', 'D'], n_rows),
        'region': np.random.choice(['North', 'South', 'East', 'West'], n_rows),
        'join_date': pd.date_range('2020-01-01', periods=n_rows, freq='D'),
        'is_premium': np.random.choice([0, 1], n_rows, p=[0.7, 0.3]),
        'email': [f'customer{i}@example.com' for i in range(n_rows)]
    }
    
    df = pd.DataFrame(data)
    
    # Introduce some data quality issues
    # Missing values
    df.loc[np.random.choice(df.index, 50), 'age'] = np.nan
    df.loc[np.random.choice(df.index, 30), 'income'] = np.nan
    
    # Duplicates
    df = pd.concat([df, df.sample(20)], ignore_index=True)
    
    # Outliers
    df.loc[np.random.choice(df.index, 10), 'income'] = np.random.uniform(200000, 500000, 10)
    
    # Save sample data
    Path('data/raw').mkdir(parents=True, exist_ok=True)
    df.to_csv('data/raw/sample_customers.csv', index=False)
    
    print(f"✓ Created sample dataset: {len(df)} rows, {len(df.columns)} columns")
    print(f"  Saved to: data/raw/sample_customers.csv")
    return df


def example_1_basic_usage():
    """Example 1: Basic pipeline usage"""
    print("\n" + "="*60)
    print("EXAMPLE 1: Basic Pipeline Usage")
    print("="*60)
    
    # Create sample data
    create_sample_data()
    
    # Initialize and run pipeline
    pipeline = DataPipeline()
    report = pipeline.run('data/raw/sample_customers.csv')
    
    print(f"\n✓ Pipeline completed in {report['duration_seconds']:.2f}s")
    print(f"✓ Processed data saved to: {report['stages']['export']['paths']['csv']}")


def example_2_custom_config():
    """Example 2: Using custom configuration"""
    print("\n" + "="*60)
    print("EXAMPLE 2: Custom Configuration")
    print("="*60)
    
    # Create custom configuration
    config = ConfigManager()
    
    # Modify settings
    config.update('cleaning.missing_values.strategy', 'fill_median')
    config.update('cleaning.outliers.action', 'remove')
    config.update('transformation.categorical_encoding.method', 'onehot')
    config.update('transformation.numerical_scaling.method', 'minmax')
    
    print("Custom settings:")
    print(f"  - Missing values: {config.get('cleaning.missing_values.strategy')}")
    print(f"  - Outliers: {config.get('cleaning.outliers.action')}")
    print(f"  - Encoding: {config.get('transformation.categorical_encoding.method')}")
    print(f"  - Scaling: {config.get('transformation.numerical_scaling.method')}")
    
    # Save custom config
    config.save('config/custom_pipeline.yaml')
    
    # Use custom config
    pipeline = DataPipeline('config/custom_pipeline.yaml')
    report = pipeline.run('data/raw/sample_customers.csv', output_name='customers_custom')
    
    print(f"\n✓ Custom pipeline completed in {report['duration_seconds']:.2f}s")


def example_3_cleaning_only():
    """Example 3: Cleaning without transformation"""
    print("\n" + "="*60)
    print("EXAMPLE 3: Cleaning Only (Skip Transformation)")
    print("="*60)
    
    pipeline = DataPipeline()
    report = pipeline.run(
        'data/raw/sample_customers.csv',
        output_name='customers_clean_only',
        skip_transformation=True
    )
    
    print(f"\n✓ Cleaning-only pipeline completed in {report['duration_seconds']:.2f}s")
    print("✓ Data is cleaned but not transformed (no encoding/scaling)")


def example_4_component_usage():
    """Example 4: Using individual components"""
    print("\n" + "="*60)
    print("EXAMPLE 4: Individual Component Usage")
    print("="*60)
    
    from src.ingestion.file_reader import FileReader
    from src.preprocessing.cleaner import DataCleaner
    from src.quality.profiler import DataProfiler
    
    # Read data
    reader = FileReader()
    df = reader.read_file('data/raw/sample_customers.csv')
    print(f"\n1. Loaded {len(df)} rows")
    
    # Profile data
    profiler = DataProfiler()
    profile = profiler.generate_profile(df)
    print(f"\n2. Data Profile:")
    print(f"   - Completeness: {profile['quality_metrics']['completeness']:.2f}%")
    print(f"   - Missing values: {profile['quality_metrics']['total_missing_values']}")
    
    # Clean data
    cleaner = DataCleaner()
    df_clean = cleaner.clean(df)
    print(f"\n3. Cleaned data: {len(df_clean)} rows (removed {len(df) - len(df_clean)} rows)")
    
    # Profile cleaned data
    profile_clean = profiler.generate_profile(df_clean)
    print(f"\n4. After Cleaning:")
    print(f"   - Completeness: {profile_clean['quality_metrics']['completeness']:.2f}%")
    print(f"   - Missing values: {profile_clean['quality_metrics']['total_missing_values']}")


def example_5_programmatic_config():
    """Example 5: Fully programmatic configuration"""
    print("\n" + "="*60)
    print("EXAMPLE 5: Programmatic Configuration")
    print("="*60)
    
    from src.preprocessing.cleaner import DataCleaner
    from src.preprocessing.transformer import DataTransformer
    
    # Read data
    df = pd.read_csv('data/raw/sample_customers.csv')
    
    # Create custom cleaner
    config = ConfigManager()
    config.update('cleaning.missing_values.strategy', 'drop')
    config.update('cleaning.duplicates.keep', 'last')
    
    cleaner = DataCleaner(config)
    df_clean = cleaner.clean(df)
    
    # Get cleaning report
    report = cleaner.get_report()
    print(f"\nCleaning actions performed: {len(report['actions'])}")
    for action in report['actions']:
        print(f"  - {action}")
    
    # Transform data
    transformer = DataTransformer(config)
    df_final = transformer.transform(df_clean)
    
    print(f"\nFinal shape: {df_final.shape}")
    print(f"Columns: {list(df_final.columns)[:10]}...")  # First 10 columns


def main():
    """Run all examples"""
    print("\n" + "🚀"*30)
    print("DATA PIPELINE USAGE EXAMPLES")
    print("🚀"*30)
    
    try:
        # Run examples
        example_1_basic_usage()
        example_2_custom_config()
        example_3_cleaning_only()
        example_4_component_usage()
        example_5_programmatic_config()
        
        print("\n" + "="*60)
        print("✓ All examples completed successfully!")
        print("="*60)
        print("\nCheck the following directories for outputs:")
        print("  - data/processed/    (processed datasets)")
        print("  - data/reports/      (quality reports)")
        print("  - logs/              (execution logs)")
        
    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()