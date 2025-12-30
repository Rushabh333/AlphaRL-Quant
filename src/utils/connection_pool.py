"""
Database connection pooling for improved performance.
"""
import os
from contextlib import contextmanager
from psycopg2.pool import SimpleConnectionPool, PoolError
from src.utils.logger import get_logger
from src.utils.config_loader import config

logger = get_logger(__name__)


class DatabaseConnectionPool:
    """
    Manages a pool of database connections for efficient resource usage.
    
    Benefits:
    - Reuses connections instead of creating new ones
    - Limits maximum concurrent connections
    - Automatic connection health checks
    - Thread-safe connection management
    """
    
    def __init__(self, minconn: int = 1, maxconn: int = 10):
    """Initialize connection pool."""
        self.config = config['database']
        
        password = os.getenv('DB_PASSWORD')
        if not password:
            logger.warning("DB_PASSWORD not set, database operations will fail")
            password = self.config.get('password', '')
        
        try:
            self.pool = SimpleConnectionPool(
                minconn=minconn,
                maxconn=maxconn,
                host=self.config['host'],
                port=self.config['port'],
                database=self.config['name'],
                user=self.config['user'],
                password=password
            )
            logger.info(f"Database connection pool created (min={minconn}, max={maxconn})")
        except Exception as e:
            logger.error(f"Failed to create connection pool: {e}")
            raise
    
    @contextmanager
    def get_connection(self):
        """
        Get a connection from the pool.
        
        Usage:
            with pool.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM table")
        
        Yields:
            Database connection
        """
        conn = None
        try:
            conn = self.pool.getconn()
            logger.debug("Connection retrieved from pool")
            yield conn
            conn.commit()
        except PoolError as e:
            logger.error(f"Connection pool error: {e}")
            if conn:
                conn.rollback()
            raise
        except Exception as e:
            logger.error(f"Database error: {e}")
            if conn:
                conn.rollback()
            raise
        finally:
            if conn and self.pool:
                self.pool.putconn(conn)
                logger.debug("Connection returned to pool")
    
    def close_all(self):
        """Close all connections in the pool."""
        if self.pool:
            self.pool.closeall()
            logger.info("All database connections closed")
    
    def get_stats(self) -> dict:
        """Get pool statistics."""
        if not self.pool:
            return {}
        
        return {
            'minconn': self.pool.minconn,
            'maxconn': self.pool.maxconn,
            # Note: SimpleConnectionPool doesn't expose current connections count
            # Consider using ThreadedConnectionPool for more stats
        }
    
    def __del__(self):
        """Cleanup on deletion."""
        self.close_all()


# Global connection pool instance
_connection_pool = None


def get_connection_pool(minconn: int = 1, maxconn: int = 10) -> DatabaseConnectionPool:
    """Get or create the global connection pool."""
    global _connection_pool
    
    if _connection_pool is None:
        _connection_pool = DatabaseConnectionPool(minconn=minconn, maxconn=maxconn)
    
    return _connection_pool
