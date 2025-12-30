"""
Main pipeline orchestrator for the ML pipeline.

This script coordinates all pipeline stages:
1. Data Collection
2. Data Processing
3. Feature Engineering
4. Data Storage
5. Model Training (future)
6. Evaluation (future)
"""
import sys
from pathlib import Path
import os

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logger import setup_logging, get_logger
from src.utils.config_loader import config
from src.utils.database import db
from src.utils.health import health_checker
from src.data.collectors import YahooFinanceCollector
from src.data.processors import DataProcessor
from src.features.engineering import FeatureEngineer
from src.utils.exceptions import PipelineException

import pandas as pd
from typing import Optional
import time

# Initialize logging
setup_logging()
logger = get_logger(__name__)


class MLPipeline:
    """
    Main pipeline orchestrator.
    Coordinates all pipeline stages with error handling and logging.
    """
    
    def __init__(self, config_dict: Optional[dict] = None):
        self.config = config_dict or config
        self.data = None
        self.processed_data = None
        self.feature_data = None
        
        logger.info("=" * 80)
        logger.info("Initializing ML Pipeline")
        logger.info("=" * 80)
    
    def run_data_collection(self) -> pd.DataFrame:
        """
        Stage 1: Data Collection
        
        Steps:
        1. Initialize collector with config
        2. Fetch data from source
        3. Validate data
        4. Return data
        """
        logger.info("\n" + "=" * 80)
        logger.info("STAGE 1: DATA COLLECTION")
        logger.info("=" * 80)
        
        try:
            collector = YahooFinanceCollector(
                tickers=self.config['data']['tickers'],
                start_date=self.config['data']['start_date'],
                end_date=self.config['data']['end_date']
            )
            
            self.data = collector.collect()
            
            logger.info(f"✓ Data collection completed: {self.data.shape}")
            return self.data
            
        except Exception as e:
            logger.error(f"✗ Data collection failed: {str(e)}")
            raise PipelineException(f"Stage 1 failed: {str(e)}") from e
    
    def run_data_processing(self) -> pd.DataFrame:
        """
        Stage 2: Data Processing
        
        Steps:
        1. Initialize processor
        2. Clean data (handle missing, outliers)
        3. Sort data
        4. Return processed data
        """
        logger.info("\n" + "=" * 80)
        logger.info("STAGE 2: DATA PROCESSING")
        logger.info("=" * 80)
        
        if self.data is None:
            raise PipelineException("No data to process. Run data collection first.")
        
        try:
            processor = DataProcessor(self.data)
            
            processing_config = {
                'missing_method': 'ffill',
                'outlier_columns': ['Volume'],
                'outlier_method': 'iqr',
                'outlier_threshold': 3.0
            }
            
            self.processed_data = processor.process(processing_config)
            
            logger.info(f"✓ Data processing completed: {self.processed_data.shape}")
            return self.processed_data
            
        except Exception as e:
            logger.error(f"✗ Data processing failed: {str(e)}")
            raise PipelineException(f"Stage 2 failed: {str(e)}") from e
    
    def run_feature_engineering(self) -> pd.DataFrame:
        """
        Stage 3: Feature Engineering
        
        Steps:
        1. Initialize feature engineer
        2. Add technical indicators
        3. Add price features
        4. Add volume features
        5. Return feature data
        """
        logger.info("\n" + "=" * 80)
        logger.info("STAGE 3: FEATURE ENGINEERING")
        logger.info("=" * 80)
        
        if self.processed_data is None:
            raise PipelineException("No processed data. Run processing first.")
        
        try:
            engineer = FeatureEngineer(self.processed_data)
            
            feature_config = self.config.get('features', {})
            
            self.feature_data = engineer.engineer_features(feature_config)
            
            logger.info(f"✓ Feature engineering completed: {self.feature_data.shape}")
            logger.info(f"  Total features: {len(engineer.feature_names)}")
            
            return self.feature_data
            
        except Exception as e:
            logger.error(f"✗ Feature engineering failed: {str(e)}")
            raise PipelineException(f"Stage 3 failed: {str(e)}") from e
    
    def save_to_database(self):
        """
        Stage 4: Data Storage
        
        Steps:
        1. Create database tables
        2. Save market data
        3. Log success
        """
        logger.info("\n" + "=" * 80)
        logger.info("STAGE 4: DATA STORAGE")
        logger.info("=" * 80)
        
        if self.feature_data is None:
            raise PipelineException("No feature data to save.")
        
        try:
            # Create tables
            db.create_tables()
            
            # Save market data
            db.save_market_data(self.processed_data)
            
            logger.info("✓ Data saved to database")
            
        except Exception as e:
            # Check if we should fail on database errors
            fail_on_db_error = self.config.get('pipeline', {}).get('fail_on_db_error', False)
            
            if fail_on_db_error:
                logger.error(f"✗ Database save failed (configured to fail): {str(e)}")
                raise
            else:
                logger.warning(f"⚠ Database save failed (continuing): {str(e)}")
                # Don't fail the pipeline if database save fails
    
    def run_full_pipeline(self, save_to_db: bool = False) -> pd.DataFrame:
        """
        Execute the complete pipeline.
        
        Steps:
        1. Stage 1: Collect data
        2. Stage 2: Process data
        3. Stage 3: Engineer features
        4. Stage 4: Save to database (optional)
        5. Log pipeline summary
        6. Return final data
        """
        start_time = time.time()
        
        logger.info("\n" + "#" * 80)
        logger.info("# STARTING FULL ML PIPELINE")
        logger.info("#" * 80 + "\n")
        
        try:
            # Stage 1: Data Collection
            self.run_data_collection()
            
            # Stage 2: Data Processing
            self.run_data_processing()
            
            # Stage 3: Feature Engineering
            self.run_feature_engineering()
            
            # Stage 4: Data Storage (optional)
            if save_to_db:
                self.save_to_database()
            
            # Pipeline summary
            elapsed_time = time.time() - start_time
            logger.info("\n" + "=" * 80)
            logger.info("PIPELINE SUMMARY")
            logger.info("=" * 80)
            logger.info(f"✓ All stages completed successfully")
            logger.info(f"  - Raw data: {self.data.shape}")
            logger.info(f"  - Processed data: {self.processed_data.shape}")
            logger.info(f"  - Feature data: {self.feature_data.shape}")
            logger.info(f"  - Total time: {elapsed_time:.2f}s")
            logger.info("=" * 80 + "\n")
            
            return self.feature_data
            
        except PipelineException as e:
            elapsed_time = time.time() - start_time
            logger.error("\n" + "=" * 80)
            logger.error("PIPELINE FAILED")
            logger.error("=" * 80)
            logger.error(f"Error: {str(e)}")
            logger.error(f"Failed after: {elapsed_time:.2f}s")
            logger.error("=" * 80 + "\n")
            raise
        
        finally:
            logger.info("Pipeline execution ended\n")



def validate_environment():
    """
    Validate required environment variables on startup.
    Optional validation - warns but doesn't fail if DB password missing.
    """
    logger.info("Validating environment variables...")
    
    # Database password is optional (pipeline can run without DB)
    if not os.getenv('DB_PASSWORD'):
        logger.warning(
            "DB_PASSWORD environment variable not set.\n"
            "Database operations will be disabled.\n"
            "Set it with: export DB_PASSWORD=your_password"
        )
    else:
        logger.info("✓ DB_PASSWORD is set")
    
    logger.info("✓ Environment validation completed")


def preflight_check():
    """
    Run health checks before starting pipeline.
    Configurable via pipeline.enable_health_checks in config.
    """
    # Check if health checks are enabled
    enable_checks = config.get('pipeline', {}).get('enable_health_checks', True)
    
    if not enable_checks:
        logger.info("Pre-flight health checks disabled in configuration")
        return
    
    logger.info("\n" + "=" * 80)
    logger.info("Running pre-flight health checks...")
    logger.info("=" * 80)
    
    status = health_checker.get_status()
    
    if not status.healthy:
        logger.warning("⚠ Some health checks failed:")
        for check, result in status.checks.items():
            if not result:
                logger.warning(f"  ✗ {check}")
        
        # Only fail on critical checks
        critical_checks = ['directories', 'environment']
        critical_failed = [name for name in critical_checks 
                          if name in status.checks and not status.checks[name]]
        
        if critical_failed:
            logger.error(f"Critical health checks failed: {critical_failed}")
            logger.error("Cannot proceed. Please fix the issues above.")
            sys.exit(1)
        else:
            logger.warning("Non-critical checks failed. Proceeding with caution...")
    else:
        logger.info("✓ All pre-flight checks passed\n")


def main():
    """Main entry point for pipeline execution."""
    
    try:
        # Pre-flight checks
        validate_environment()
        preflight_check()
        
        # Initialize and run pipeline
        pipeline = MLPipeline()
        feature_data = pipeline.run_full_pipeline(save_to_db=False)
        
        # Save to CSV for inspection
        output_path = Path("data/processed/features.csv")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        feature_data.to_csv(output_path, index=False)
        logger.info(f"Feature data saved to {output_path}")
        
        print("\n✓ Pipeline completed successfully!")
        print(f"  Output: {output_path}")
        print(f"  Shape: {feature_data.shape}")
        
    except Exception as e:
        print(f"\n✗ Pipeline failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
