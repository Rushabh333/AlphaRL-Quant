"""
AlphaRL-Quant Prometheus Metrics
Exposes application metrics for monitoring and alerting.
"""

from prometheus_client import (
    Counter, Gauge, Histogram, Summary, Info,
    CollectorRegistry, generate_latest, start_http_server,
    CONTENT_TYPE_LATEST
)
from prometheus_client import multiprocess, CollectorRegistry
from typing import Optional
import time
import functools
import os


# Create registry
registry = CollectorRegistry()

# =============================================================================
# Pipeline Metrics
# =============================================================================

PIPELINE_RUNS_TOTAL = Counter(
    'pipeline_runs_total',
    'Total number of pipeline runs',
    ['status'],  # success, failed
    registry=registry
)

PIPELINE_DURATION_SECONDS = Histogram(
    'pipeline_duration_seconds',
    'Time spent running pipeline',
    ['stage'],  # collection, processing, feature_engineering
    buckets=[1, 5, 10, 30, 60, 120, 300, 600],
    registry=registry
)

PIPELINE_ROWS_PROCESSED = Counter(
    'pipeline_rows_processed_total',
    'Total number of data rows processed',
    ['ticker'],
    registry=registry
)

PIPELINE_ERRORS = Counter(
    'pipeline_errors_total',
    'Total pipeline errors',
    ['error_type'],
    registry=registry
)

# =============================================================================
# Data Collection Metrics
# =============================================================================

API_REQUESTS_TOTAL = Counter(
    'api_requests_total',
    'Total API requests made',
    ['source', 'status'],  # source: yahoo, alpha_vantage; status: success, failed
    registry=registry
)

API_REQUEST_DURATION = Histogram(
    'api_request_duration_seconds',
    'API request latency',
    ['source'],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
    registry=registry
)

API_RATE_LIMIT_HITS = Counter(
    'api_rate_limit_hits_total',
    'Number of times rate limit was hit',
    ['source'],
    registry=registry
)

DATA_CACHE_HITS = Counter(
    'data_cache_hits_total',
    'Cache hits for data requests',
    ['cache_type'],  # memory, disk, database
    registry=registry
)

DATA_CACHE_MISSES = Counter(
    'data_cache_misses_total',
    'Cache misses for data requests',
    ['cache_type'],
    registry=registry
)

# =============================================================================
# Feature Engineering Metrics
# =============================================================================

FEATURES_GENERATED = Counter(
    'features_generated_total',
    'Total number of features generated',
    ['feature_type'],  # technical_indicator, lag, derived
    registry=registry
)

FEATURE_COMPUTATION_TIME = Histogram(
    'feature_computation_seconds',
    'Time to compute features',
    ['feature_name'],
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 5.0],
    registry=registry
)

FEATURE_NULL_PERCENTAGE = Gauge(
    'feature_null_percentage',
    'Percentage of null values in feature',
    ['feature_name'],
    registry=registry
)

# =============================================================================
# Training Metrics
# =============================================================================

TRAINING_EPISODES = Counter(
    'training_episodes_total',
    'Total training episodes completed',
    registry=registry
)

TRAINING_TIMESTEPS = Counter(
    'training_timesteps_total',
    'Total training timesteps',
    registry=registry
)

EPISODE_REWARD = Gauge(
    'episode_reward',
    'Current episode reward',
    registry=registry
)

EPISODE_LENGTH = Gauge(
    'episode_length',
    'Current episode length',
    registry=registry
)

PORTFOLIO_VALUE = Gauge(
    'portfolio_value_usd',
    'Current portfolio value in USD',
    ['agent'],
    registry=registry
)

SHARPE_RATIO = Gauge(
    'sharpe_ratio',
    'Current Sharpe ratio',
    ['agent'],
    registry=registry
)

MAX_DRAWDOWN = Gauge(
    'max_drawdown',
    'Maximum drawdown percentage',
    ['agent'],
    registry=registry
)

MODEL_CHECKPOINT_SAVED = Counter(
    'model_checkpoints_saved_total',
    'Number of model checkpoints saved',
    registry=registry
)

# =============================================================================
# Database Metrics
# =============================================================================

DB_QUERIES_TOTAL = Counter(
    'db_queries_total',
    'Total database queries',
    ['operation', 'table'],  # operation: SELECT, INSERT, UPDATE
    registry=registry
)

DB_QUERY_DURATION = Histogram(
    'db_query_duration_seconds',
    'Database query execution time',
    ['operation'],
    buckets=[0.001, 0.01, 0.1, 0.5, 1.0, 5.0],
    registry=registry
)

DB_CONNECTION_POOL_SIZE = Gauge(
    'db_connection_pool_size',
    'Current database connection pool size',
    registry=registry
)

DB_CONNECTIONS_ACTIVE = Gauge(
    'db_connections_active',
    'Number of active database connections',
    registry=registry
)

DB_ERRORS = Counter(
    'db_errors_total',
    'Database errors',
    ['error_type'],
    registry=registry
)

# =============================================================================
# System Metrics
# =============================================================================

SYSTEM_INFO = Info(
    'alpharl_system',
    'AlphaRL-Quant system information',
    registry=registry
)

HEALTH_STATUS = Gauge(
    'health_status',
    'System health status (1=healthy, 0=unhealthy)',
    ['component'],  # database, api, filesystem
    registry=registry
)

# =============================================================================
# Metric Helper Functions
# =============================================================================

def track_pipeline_run(status: str):
    """Track a pipeline run."""
    PIPELINE_RUNS_TOTAL.labels(status=status).inc()


def track_api_request(source: str, status: str, duration: float):
    """Track an API request."""
    API_REQUESTS_TOTAL.labels(source=source, status=status).inc()
    API_REQUEST_DURATION.labels(source=source).observe(duration)


def track_feature_computation(feature_name: str, duration: float):
    """Track feature computation time."""
    FEATURE_COMPUTATION_TIME.labels(feature_name=feature_name).observe(duration)


def track_db_query(operation: str, table: str, duration: float):
    """Track a database query."""
    DB_QUERIES_TOTAL.labels(operation=operation, table=table).inc()
    DB_QUERY_DURATION.labels(operation=operation).observe(duration)


def update_portfolio_metrics(agent: str, value: float, sharpe: float, drawdown: float):
    """Update portfolio performance metrics."""
    PORTFOLIO_VALUE.labels(agent=agent).set(value)
    SHARPE_RATIO.labels(agent=agent).set(sharpe)
    MAX_DRAWDOWN.labels(agent=agent).set(drawdown)


def set_health_status(component: str, is_healthy: bool):
    """Set component health status."""
    HEALTH_STATUS.labels(component=component).set(1.0 if is_healthy else 0.0)


# =============================================================================
# Decorators for Automatic Metric Tracking
# =============================================================================

def track_time(metric: Histogram, labels: Optional[dict] = None):
    """
    Decorator to track execution time of a function.
    
    Example:
        >>> @track_time(PIPELINE_DURATION_SECONDS, labels={'stage': 'collection'})
        >>> def collect_data():
        >>>     pass
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start = time.time()
            try:
                return func(*args, **kwargs)
            finally:
                duration = time.time() - start
                if labels:
                    metric.labels(**labels).observe(duration)
                else:
                    metric.observe(duration)
        return wrapper
    return decorator


def count_calls(counter: Counter, labels: Optional[dict] = None):
    """
    Decorator to count function calls.
    
    Example:
        >>> @count_calls(TRAINING_EPISODES)
        >>> def train_episode():
        >>>     pass
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                result = func(*args, **kwargs)
                if labels:
                    counter.labels(**labels).inc()
                else:
                    counter.inc()
                return result
            except Exception as e:
                if labels:
                    counter.labels(**labels, status='error').inc()
                raise
        return wrapper
    return decorator


# =============================================================================
# Metrics Server
# =============================================================================

def start_metrics_server(port: int = 9090):
    """
    Start Prometheus metrics HTTP server.
    
    Args:
        port: Port to listen on (default: 9090)
    
    Example:
        >>> start_metrics_server(port=9090)
        >>> # Metrics available at http://localhost:9090/metrics
    """
    start_http_server(port, registry=registry)
    print(f"✅ Metrics server started on port {port}")
    print(f"📊 Metrics endpoint: http://localhost:{port}/metrics")


def get_metrics() -> bytes:
    """
    Get current metrics in Prometheus format.
    Returns bytes suitable for HTTP response.
    """
    return generate_latest(registry)


# =============================================================================
# Initialization
# =============================================================================

def init_metrics():
    """Initialize metrics with system information."""
    SYSTEM_INFO.info({
        'version': '1.0.0',
        'python_version': os.sys.version.split()[0],
        'environment': os.getenv('ENVIRONMENT', 'development')
    })
    
    # Initialize health gauges
    for component in ['database', 'api', 'filesystem']:
        HEALTH_STATUS.labels(component=component).set(0.0)


# Auto-initialize on import
init_metrics()


# =============================================================================
# Example Usage
# =============================================================================

if __name__ == "__main__":
    # Start metrics server
    start_metrics_server(port=9090)
    
    # Simulate some metrics
    import random
    import time
    
    print("Generating sample metrics...")
    
    for i in range(100):
        # Pipeline metrics
        track_pipeline_run(status=random.choice(['success', 'failed']))
        
        # API metrics
        track_api_request(
            source='yahoo',
            status='success',
            duration=random.uniform(0.1, 2.0)
        )
        
        # Portfolio metrics
        update_portfolio_metrics(
            agent='ppo',
            value=10000 + random.uniform(-1000, 1000),
            sharpe=random.uniform(0.5, 2.0),
            drawdown=random.uniform(0, 20)
        )
        
        # Health status
        set_health_status('database', random.choice([True, False]))
        
        time.sleep(0.1)
    
    print(f"\n✅ Metrics generated")
    print(f"📊 View metrics at: http://localhost:9090/metrics")
    print("Press Ctrl+C to stop")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping metrics server...")
