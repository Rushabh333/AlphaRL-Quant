#!/usr/bin/env python3
"""
Environment Configuration Validator

Ensures environment configurations are safe for deployment and prevents
common production mistakes.

Usage:
    python scripts/validate_env.py --env production
    python scripts/validate_env.py --env staging
"""

import argparse
import sys
import yaml
from pathlib import Path
from typing import Dict, List, Any
from dataclasses import dataclass

# Colors for output
RED = '\033[0;31m'
GREEN = '\033[0;32m'
YELLOW = '\033[1;33m'
BLUE = '\033[0;34m'
NC = '\033[0m'  # No Color


@dataclass
class ValidationResult:
    """Result of a validation check"""
    passed: bool
    message: str
    severity: str = "error"  # error, warning, info


class EnvironmentValidator:
    """Validates environment configurations"""
    
    def __init__(self, env_name: str):
        self.env_name = env_name
        self.config_dir = Path("config")
        self.errors: List[ValidationResult] = []
        self.warnings: List[ValidationResult] = []
        self.config: Dict[str, Any] = {}
    
    def load_config(self) -> bool:
        """Load and merge base + environment config"""
        try:
            # Load base config
            base_path = self.config_dir / "base.yaml"
            with open(base_path) as f:
                self.config = yaml.safe_load(f)
            
            # Load environment config
            env_path = self.config_dir / "environments" / f"{self.env_name}.yaml"
            if not env_path.exists():
                self.errors.append(ValidationResult(
                    False,
                    f"Environment config not found: {env_path}",
                    "error"
                ))
                return False
            
            with open(env_path) as f:
                env_config = yaml.safe_load(f)
            
            # Deep merge configs (env overrides base)
            self._deep_merge(self.config, env_config)
            
            return True
            
        except Exception as e:
            self.errors.append(ValidationResult(
                False,
                f"Failed to load config: {str(e)}",
                "error"
            ))
            return False
    
    def _deep_merge(self, base: Dict, override: Dict) -> None:
        """Deep merge override dict into base dict"""
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = value
    
    def validate_production(self) -> List[ValidationResult]:
        """Strictly validate production environment"""
        results = []
        
        # Get feature flags
        flags = self.config.get('feature_flags', {})
        
        # CRITICAL: Debug mode MUST be false
        if flags.get('debug_mode', False):
            results.append(ValidationResult(
                False,
                "⛔ CRITICAL: debug_mode is enabled in production!",
                "error"
            ))
        
        # CRITICAL: Mock data MUST be false
        if flags.get('use_mock_data', False):
            results.append(ValidationResult(
                False,
                "⛔ CRITICAL: use_mock_data is enabled in production!",
                "error"
            ))
        
        # CRITICAL: Strict validation MUST be true
        if not flags.get('strict_validation', True):
            results.append(ValidationResult(
                False,
                "⛔ CRITICAL: strict_validation is disabled in production!",
                "error"
            ))
        
        # CRITICAL: Circuit breaker MUST be enabled
        if not flags.get('circuit_breaker', False):
            results.append(ValidationResult(
                False,
                "⛔ CRITICAL: circuit_breaker is disabled in production!",
                "error"
            ))
        
        # Database SSL
        db = self.config.get('database', {})
        if not db.get('ssl_required', False):
            results.append(ValidationResult(
                False,
                "⚠️ WARNING: Database SSL not enforced in production",
                "warning"
            ))
        
        # API rate limiting
        api = self.config.get('api', {})
        if not api.get('rate_limit', {}).get('enabled', False):
            results.append(ValidationResult(
                False,
                "⚠️ WARNING: API rate limiting not enabled",
                "warning"
            ))
        
        # Trading limits
        trading = self.config.get('trading', {})
        limits = trading.get('limits', {})
        
        if not limits.get('max_drawdown_percent'):
            results.append(ValidationResult(
                False,
                "⚠️ WARNING: No max_drawdown_percent limit set",
                "warning"
            ))
        
        if not limits.get('max_trades_per_day'):
            results.append(ValidationResult(
                False,
                "⚠️ WARNING: No max_trades_per_day limit set",
                "warning"
            ))
        
        # Logging
        logging = self.config.get('logging', {})
        if logging.get('level') == 'DEBUG':
            results.append(ValidationResult(
                False,
                "⚠️ WARNING: Logging level is DEBUG in production (performance impact)",
                "warning"
            ))
        
        if logging.get('format') != 'json':
            results.append(ValidationResult(
                False,
                "⚠️ WARNING: Logging format should be 'json' for production",
                "warning"
            ))
        
        # Backup configuration
        backup = self.config.get('backup', {})
        if not backup.get('enabled', False):
            results.append(ValidationResult(
                False,
                "⚠️ WARNING: Backups are not enabled in production",
                "warning"
            ))
        
        # Monitoring
        monitoring = self.config.get('monitoring', {})
        if not monitoring.get('alerts', {}).get('enabled', False):
            results.append(ValidationResult(
                False,
                "⚠️ WARNING: Alerts are not enabled in production",
                "warning"
            ))
        
        # Check for environment variables
        if '${' in str(self.config):
            results.append(ValidationResult(
                True,
                "ℹ️ INFO: Config contains environment variable placeholders (ensure they're set)",
                "info"
            ))
        
        return results
    
    def validate_staging(self) -> List[ValidationResult]:
        """Validate staging environment"""
        results = []
        
        flags = self.config.get('feature_flags', {})
        
        # Paper trading should be enabled
        if not flags.get('enable_paper_trading', True):
            results.append(ValidationResult(
                False,
                "⚠️ WARNING: Paper trading disabled in staging (recommend enabling)",
                "warning"
            ))
        
        # Strict validation should be true
        if not flags.get('strict_validation', True):
            results.append(ValidationResult(
                False,
                "⚠️ WARNING: Strict validation should be enabled in staging",
                "warning"
            ))
        
        return results
    
    def validate_development(self) -> List[ValidationResult]:
        """Validate development environment"""
        results = []
        
        # Just informational checks for dev
        flags = self.config.get('feature_flags', {})
        
        if not flags.get('use_mock_data', False):
            results.append(ValidationResult(
                True,
                "ℹ️ INFO: Mock data disabled - will use real APIs",
                "info"
            ))
        
        return results
    
    def validate(self) -> bool:
        """Run all validations for the environment"""
        if not self.load_config():
            return False
        
        # Run environment-specific validations
        if self.env_name == "production":
            results = self.validate_production()
        elif self.env_name == "staging":
            results = self.validate_staging()
        elif self.env_name == "development":
            results = self.validate_development()
        else:
            self.errors.append(ValidationResult(
                False,
                f"Unknown environment: {self.env_name}",
                "error"
            ))
            return False
        
        # Categorize results
        for result in results:
            if result.severity == "error":
                self.errors.append(result)
            elif result.severity == "warning":
                self.warnings.append(result)
        
        # Return success if no errors
        return len(self.errors) == 0
    
    def print_results(self):
        """Print validation results"""
        print(f"\n{BLUE}╔══════════════════════════════════════════════════════╗{NC}")
        print(f"{BLUE}║  Environment Configuration Validator                ║{NC}")
        print(f"{BLUE}╚══════════════════════════════════════════════════════╝{NC}\n")
        
        print(f"Environment: {BLUE}{self.env_name}{NC}\n")
        
        # Print errors
        if self.errors:
            print(f"{RED}❌ ERRORS ({len(self.errors)}):{NC}")
            for error in self.errors:
                print(f"  {RED}•{NC} {error.message}")
            print()
        
        # Print warnings
        if self.warnings:
            print(f"{YELLOW}⚠️  WARNINGS ({len(self.warnings)}):{NC}")
            for warning in self.warnings:
                print(f"  {YELLOW}•{NC} {warning.message}")
            print()
        
        # Summary
        if not self.errors and not self.warnings:
            print(f"{GREEN}✅ ALL VALIDATIONS PASSED{NC}")
            print(f"{GREEN}Configuration is safe for {self.env_name} deployment!{NC}\n")
            return True
        elif not self.errors:
            print(f"{GREEN}✅ NO ERRORS{NC}")
            print(f"Configuration passes all checks (with {len(self.warnings)} warnings)\n")
            return True
        else:
            print(f"{RED}❌ VALIDATION FAILED{NC}")
            print(f"{RED}Fix {len(self.errors)} error(s) before deploying to {self.env_name}!{NC}\n")
            return False


def main():
    parser = argparse.ArgumentParser(description="Validate environment configuration")
    parser.add_argument(
        '--env',
        required=True,
        choices=['development', 'staging', 'production'],
        help='Environment to validate'
    )
    
    args = parser.parse_args()
    
    validator = EnvironmentValidator(args.env)
    
    if validator.validate():
        validator.print_results()
        sys.exit(0)
    else:
        validator.print_results()
        sys.exit(1)


if __name__ == "__main__":
    main()
