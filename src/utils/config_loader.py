"""
Centralized configuration management for the ML pipeline.
Supports multi-environment configurations with base + environment overrides.
"""
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import os
import logging

logger = logging.getLogger(__name__)


class ConfigLoader:
    """
    Multi-environment configuration management.
    
    Loads base.yaml and merges with environment-specific config (dev/staging/prod).
    Handles environment variable substitution and provides convenience methods.
    
    Usage:
        config = ConfigLoader()
        db_host = config.get('database.host')
        mock_data = config.feature_flag('use_mock_data')
    """
    
    def __init__(self, environment: Optional[str] = None):
        """
        Initialize config loader.
        
        Args:
            environment: Environment name (development, staging, production).
                        If None, uses ENVIRONMENT env var or defaults to development.
        """
        self.environment = environment or os.getenv('ENVIRONMENT', 'development')
        self.config_dir = Path("config")
        self._config = None
        self._feature_flags = None
    
    @property
    def config(self) -> Dict[str, Any]:
        """Lazy load configuration."""
        if self._config is None:
            self._config = self._load_config()
        return self._config
    
    @property
    def feature_flags(self) -> Dict[str, Any]:
        """Get feature flags for current environment."""
        if self._feature_flags is None:
            self._feature_flags = self.config.get('feature_flags', {})
        return self._feature_flags
    
    def _load_config(self) -> Dict[str, Any]:
        """Load and merge base + environment configurations."""
        # Load base config
        base_path = self.config_dir / "base.yaml"
        if not base_path.exists():
            raise FileNotFoundError(f"Base config not found: {base_path}")
        
        with open(base_path, 'r') as f:
            config = yaml.safe_load(f) or {}
        
        # Load environment-specific config
        env_path = self.config_dir / "environments" / f"{self.environment}.yaml"
        if env_path.exists():
            with open(env_path, 'r') as f:
                env_config = yaml.safe_load(f) or {}
            
            # Deep merge environment config into base
            config = self._deep_merge(config, env_config)
            logger.info(f"Loaded configuration for environment: {self.environment}")
        else:
            logger.warning(f"Environment config not found: {env_path}, using base only")
        
        # Replace environment variables
        config = self._replace_env_vars(config)
        
        # Add metadata
        config['_environment'] = self.environment
        config['_loaded_at'] = __import__('datetime').datetime.utcnow().isoformat()
        
        return config
    
    def _deep_merge(self, base: Dict, override: Dict) -> Dict:
        """
        Deep merge override dict into base dict.
        
        override values take precedence over base values.
        Nested dicts are merged recursively.
        """
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def _replace_env_vars(self, config: Any) -> Any:
        """
        Replace ${ENV_VAR} with actual environment variables.
        
        Recursively processes dicts and lists.
        Falls back to original value if env var not set.
        """
        if isinstance(config, dict):
            return {k: self._replace_env_vars(v) for k, v in config.items()}
        elif isinstance(config, list):
            return [self._replace_env_vars(item) for item in config]
        elif isinstance(config, str) and config.startswith("${") and config.endswith("}"):
            env_var = config[2:-1]
            value = os.getenv(env_var)
            if value is None:
                logger.warning(f"Environment variable not set: {env_var}, using placeholder")
                return config
            return value
        return config
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by dot-notation key.
        
        Args:
            key: Dot-separated key path (e.g., 'database.host')
            default: Default value if key not found
        
        Returns:
            Configuration value or default
        
        Example:
            db_host = config.get('database.host', 'localhost')
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default
        
        return value if value is not None else default
    
    def feature_flag(self, flag_name: str, default: bool = False) -> bool:
        """
        Get feature flag value.
        
        Args:
            flag_name: Name of the feature flag
            default: Default value if flag not found
        
        Returns:
            Boolean flag value
        
        Example:
            if config.feature_flag('use_mock_data'):
                return MockDataSource()
        """
        return self.feature_flags.get(flag_name, default)
    
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment == 'production'
    
    def is_staging(self) -> bool:
        """Check if running in staging environment."""
        return self.environment == 'staging'
    
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.environment == 'development'
    
    def require(self, key: str) -> Any:
        """
        Get required configuration value, raise error if not found.
        
        Args:
            key: Dot-separated key path
        
        Returns:
            Configuration value
        
        Raises:
            KeyError: If key not found
        
        Example:
            db_host = config.require('database.host')
        """
        value = self.get(key)
        if value is None:
            raise KeyError(f"Required configuration key not found: {key}")
        return value
    
    def reload(self) -> None:
        """Reload configuration from files."""
        self._config = None
        self._feature_flags = None
        logger.info("Configuration reloaded")


# Global config instance
_global_config = None


def get_config(environment: Optional[str] = None) -> ConfigLoader:
    """
    Get global configuration instance.
    
    Args:
        environment: Optional environment override
    
    Returns:
        ConfigLoader instance
    """
    global _global_config
    if _global_config is None or environment is not None:
        _global_config = ConfigLoader(environment)
    return _global_config


# Convenience shortcuts
config = get_config().config  # For backward compatibility

