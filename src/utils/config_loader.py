"""
Centralized configuration management for the ML pipeline.
"""
import yaml
from pathlib import Path
from typing import Dict, Any
import os


class ConfigLoader:
    """
    Centralized configuration management.
    Loads configuration from YAML file and handles environment variable substitution.
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config_path = Path(config_path)
        self._config = None
    
    @property
    def config(self) -> Dict[str, Any]:
        """Lazy load configuration."""
        if self._config is None:
            self._config = self._load_config()
        return self._config
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Replace environment variables
        config = self._replace_env_vars(config)
        return config
    
    def _replace_env_vars(self, config: Dict) -> Dict:
        """Replace ${ENV_VAR} with actual environment variables."""
        if isinstance(config, dict):
            return {k: self._replace_env_vars(v) for k, v in config.items()}
        elif isinstance(config, list):
            return [self._replace_env_vars(item) for item in config]
        elif isinstance(config, str) and config.startswith("${") and config.endswith("}"):
            env_var = config[2:-1]
            return os.getenv(env_var, config)
        return config
    
    def get(self, key: str, default: Any = None) -> Any:
    """Get configuration value by dot-notation key."""
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k, default)
            else:
                return default
        
        return value


# Global config instance
config = ConfigLoader().config
