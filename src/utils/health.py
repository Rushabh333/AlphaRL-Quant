"""
Health check system for monitoring pipeline components.
"""
from dataclasses import dataclass, asdict
from typing import Dict
import time
import logging
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class HealthStatus:
    """Health status for the pipeline."""
    healthy: bool
    checks: Dict[str, bool]
    timestamp: float
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
    
    def __str__(self) -> str:
        """Human-readable status."""
        status = "✅ HEALTHY" if self.healthy else "❌ UNHEALTHY"
        checks_str = "\n".join(
            f"  {'✅' if v else '❌'} {k}" 
            for k, v in self.checks.items()
        )
        return f"{status}\n{checks_str}"


class HealthChecker:
    """
    Check health of pipeline components.
    
    Used for monitoring and pre-flight checks before running pipeline.
    """
    
    def check_database(self) -> bool:
        """
        Check if database is accessible.
        
        Returns:
            True if database is healthy, False otherwise
        """
        try:
            from src.utils.database import db
            return db.health_check()
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return False
    
    def check_yahoo_finance(self) -> bool:
        """
        Check if Yahoo Finance API is accessible.
        
        Returns:
            True if Yahoo Finance is accessible, False otherwise
        """
        try:
            import yfinance as yf
            # Try to get basic info for a known ticker
            ticker = yf.Ticker("AAPL")
            info = ticker.info
            
            # Verify we got actual data
            if info and 'symbol' in info:
                logger.debug("Yahoo Finance API is accessible")
                return True
            else:
                logger.warning("Yahoo Finance returned empty data")
                return False
                
        except Exception as e:
            logger.error(f"Yahoo Finance health check failed: {e}")
            return False
    
    def check_required_directories(self) -> bool:
        """
        Check if required directories exist.
        
        Returns:
            True if all required directories exist, False otherwise
        """
        try:
            from pathlib import Path
            required_dirs = ['data/cache', 'data/processed', 'logs', 'checkpoints']
            
            for dir_path in required_dirs:
                path = Path(dir_path)
                if not path.exists():
                    logger.warning(f"Required directory missing: {dir_path}")
                    path.mkdir(parents=True, exist_ok=True)
                    logger.info(f"Created directory: {dir_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"Directory check failed: {e}")
            return False
    
    def check_environment_variables(self) -> bool:
        """
        Check if required environment variables are set.
        
        Returns:
            True if all required variables are set, False otherwise
        """
        try:
            import os
            # Optional variables that should be set for database operations
            optional_vars = ['DB_PASSWORD']
            
            missing = [var for var in optional_vars if not os.getenv(var)]
            
            if missing:
                logger.warning(
                    f"Optional environment variables not set: {missing}. "
                    f"Database operations will be disabled."
                )
                # Don't fail, just warn
            
            return True
            
        except Exception as e:
            logger.error(f"Environment variable check failed: {e}")
            return False
    
    def get_status(self) -> HealthStatus:
        """
        Get overall health status.
        
        Returns:
            HealthStatus object with all check results
        """
        logger.info("Running health checks...")
        
        checks = {
            'directories': self.check_required_directories(),
            'environment': self.check_environment_variables(),
            'yahoo_finance': self.check_yahoo_finance(),
            'database': self.check_database(),
        }
        
        healthy = all(checks.values())
        
        status = HealthStatus(
            healthy=healthy,
            checks=checks,
            timestamp=time.time()
        )
        
        if healthy:
            logger.info("✓ All health checks passed")
        else:
            failed = [name for name, result in checks.items() if not result]
            logger.warning(f"Health checks failed: {failed}")
        
        return status


# Global health checker instance
health_checker = HealthChecker()
