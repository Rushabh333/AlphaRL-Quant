"""
Database management for the ML pipeline.
Now with connection pooling for better performance.
"""
import os
import psycopg2
from psycopg2.extras import execute_values
import pandas as pd
from typing import Optional
from src.utils.logger import get_logger
from src.utils.connection_pool import get_connection_pool

logger =get_logger(__name__)


class DatabaseManager:
    """
    Manages database connections and operations.
    Uses connection pooling for efficient resource usage.
    Pool is created lazily on first use to allow running without database.
    """
    
    def __init__(self):
        """Initialize DatabaseManager with lazy pool creation."""
        # Validate password is set
        if not os.getenv('DB_PASSWORD'):
            logger.warning(
                "DB_PASSWORD environment variable not set. "
                "Database operations will be disabled. "
                "Set it with: export DB_PASSWORD=your_password"
            )
        
        # Don't initialize pool immediately - do it lazily
        self._pool = None
        logger.info("Database manager initialized (pool will be created on first use)")
    
    @property
    def pool(self):
        """Get connection pool, creating it if necessary."""
        if self._pool is None:
            try:
                self._pool = get_connection_pool(minconn=1, maxconn=10)
                logger.info("Database connection pool created")
            except Exception as e:
                logger.error(f"Failed to create database pool: {e}")
                logger.error("Database operations will be disabled")
                raise
        return self._pool
    
    def get_connection(self):
        """
        Get a connection from the pool.
        
        Usage:
            with db.get_connection() as conn:
                cursor = conn.cursor()
                # Use connection
        
        Yields:
            Database connection from pool
        """
        return self.pool.get_connection()
    
    def health_check(self) -> bool:
        """
        Check if database connection is healthy.
        
        Returns:
            True if connection is healthy, False otherwise
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT 1")
                result = cursor.fetchone()
                return result == (1,)
        except Exception as e:
            logger.error(f"Database health check failed: {str(e)}")
            return False
    
    def create_tables(self):
        """Create necessary tables if they don't exist."""
        
        create_table_queries = [
            """
            CREATE TABLE IF NOT EXISTS market_data (
                id SERIAL PRIMARY KEY,
                ticker VARCHAR(10) NOT NULL,
                date TIMESTAMP NOT NULL,
                open NUMERIC(12, 4),
                high NUMERIC(12, 4),
                low NUMERIC(12, 4),
                close NUMERIC(12, 4),
                volume BIGINT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(ticker, date)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS features (
                id SERIAL PRIMARY KEY,
                ticker VARCHAR(10) NOT NULL,
                date TIMESTAMP NOT NULL,
                feature_name VARCHAR(50) NOT NULL,
                feature_value NUMERIC(12, 6),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS training_runs (
                id SERIAL PRIMARY KEY,
                run_name VARCHAR(100) NOT NULL,
                algorithm VARCHAR(50),
                total_timesteps INTEGER,
                final_reward NUMERIC(12, 6),
                config JSONB,
                started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                completed_at TIMESTAMP
            )
            """
        ]
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            for query in create_table_queries:
                cursor.execute(query)
            logger.info("Database tables created/verified")
    
    def save_market_data(self, df: pd.DataFrame):
        """
        Save market data to database.
        Uses batch insert for efficiency.
        """
        logger.info(f"Saving {len(df)} rows of market data")
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Prepare data
            columns = ['ticker', 'Date', 'Open', 'High', 'Low', 'Close', 'Volume']
            # Check which columns exist
            available_cols = [col for col in columns if col in df.columns]
            values = df[available_cols].values.tolist()
            
            # Batch insert with conflict handling
            query = f"""
                INSERT INTO market_data (ticker, date, open, high, low, close, volume)
                VALUES %s
                ON CONFLICT (ticker, date) DO UPDATE SET
                    open = EXCLUDED.open,
                    high = EXCLUDED.high,
                    low = EXCLUDED.low,
                    close = EXCLUDED.close,
                    volume = EXCLUDED.volume
            """
            
            execute_values(cursor, query, values)
            logger.info(f"Saved {len(df)} rows to database")
    
    def load_market_data(self, ticker: Optional[str] = None,
                         start_date: Optional[str] = None,
                         end_date: Optional[str] = None) -> pd.DataFrame:
        """Load market data from database with optional filters."""
        
        query = "SELECT * FROM market_data WHERE 1=1"
        params = []
        
        if ticker:
            query += " AND ticker = %s"
            params.append(ticker)
        
        if start_date:
            query += " AND date >= %s"
            params.append(start_date)
        
        if end_date:
            query += " AND date <= %s"
            params.append(end_date)
        
        query += " ORDER BY ticker, date"
        
        with self.get_connection() as conn:
            df = pd.read_sql_query(query, conn, params=params)
        
        logger.info(f"Loaded {len(df)} rows from database")
        return df


# Global database instance
db = DatabaseManager()
