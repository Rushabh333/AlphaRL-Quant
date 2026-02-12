#!/usr/bin/env python3
"""
AlphaRL-Quant Secrets Validation
Validates that all required secrets and configurations are properly set.
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import re

# ANSI color codes
GREEN = '\033[92m'
YELLOW = '\033[93m'
RED = '\033[91m'
RESET = '\033[0m'
BOLD = '\033[1m'


class SecretsValidator:
    """Validate environment variables and secrets."""
    
    # Required environment variables
    REQUIRED_VARS = [
        "POSTGRES_PASSWORD",
        "SECRET_KEY",
        "JWT_SECRET"
    ]
    
    # Optional but recommended variables
    RECOMMENDED_VARS = [
        "YAHOO_FINANCE_API_KEY",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "ENCRYPTION_KEY"
    ]
    
    # Variables that should be production-ready
    PRODUCTION_VARS = {
        "ENVIRONMENT": ["production", "staging", "development"],
        "LOG_LEVEL": ["DEBUG", "INFO", "WARNING", "ERROR"],
        "POSTGRES_USER": None,  # Any value OK
        "POSTGRES_DB": None
    }
    
    def __init__(self):
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.info: List[str] = []
        
    def check_env_file_exists(self) -> bool:
        """Check if .env file exists."""
        env_path = Path(".env")
        if not env_path.exists():
            self.errors.append(".env file not found")
            return False
        
        # Check permissions (should be 600)
        stat_info = env_path.stat()
        permissions = oct(stat_info.st_mode)[-3:]
        
        if permissions != "600":
            self.warnings.append(
                f".env has permissions {permissions} (should be 600)\n"
                f"  Fix with: chmod 600 .env"
            )
        
        self.info.append(f"✓ .env file exists with permissions {permissions}")
        return True
    
    def check_gitignore(self) -> bool:
        """Check if .env is in .gitignore."""
        gitignore_path = Path(".gitignore")
        
        if not gitignore_path.exists():
            self.warnings.append(".gitignore not found")
            return False
        
        content = gitignore_path.read_text()
        
        # Check for .env patterns
        env_patterns = [r'\.env$', r'\.env\b', r'^\.env', r'\.secrets']
        found = any(re.search(pattern, content, re.MULTILINE) 
                   for pattern in env_patterns)
        
        if not found:
            self.errors.append(
                ".env not found in .gitignore - SECURITY RISK!\n"
                "  Add: echo '.env' >> .gitignore"
            )
            return False
        
        self.info.append("✓ .env is in .gitignore")
        return True
    
    def validate_required_vars(self) -> bool:
        """Validate all required environment variables."""
        missing = []
        
        for var in self.REQUIRED_VARS:
            value = os.getenv(var)
            if not value:
                missing.append(var)
            else:
                # Validate specific formats
                if var in ["SECRET_KEY", "JWT_SECRET", "ENCRYPTION_KEY"]:
                    if len(value) < 32:
                        self.warnings.append(
                            f"{var} is too short ({len(value)} chars, recommended: 64+)"
                        )
                    else:
                        self.info.append(f"✓ {var} is set ({len(value)} chars)")
                
                elif var == "POSTGRES_PASSWORD":
                    if len(value) < 8:
                        self.warnings.append(
                            f"{var} is weak (length: {len(value)}, min: 8)"
                        )
                    elif value in ["changeme", "password", "admin", "postgres"]:
                        self.errors.append(
                            f"{var} uses a common/default password - CHANGE IT!"
                        )
                    else:
                        self.info.append(f"✓ {var} is set")
        
        if missing:
            self.errors.append(
                f"Missing required variables: {', '.join(missing)}\n"
                f"  Run: bash scripts/setup_secrets.sh"
            )
            return False
        
        return True
    
    def validate_recommended_vars(self) -> bool:
        """Check optional but recommended variables."""
        missing = []
        
        for var in self.RECOMMENDED_VARS:
            value = os.getenv(var)
            if not value:
                missing.append(var)
            else:
                self.info.append(f"✓ {var} is configured")
        
        if missing:
            self.warnings.append(
                f"Optional variables not set: {', '.join(missing)}\n"
                f"  (Non-critical, but recommended for production)"
            )
        
        return True
    
    def validate_production_vars(self) -> bool:
        """Validate production-specific configurations."""
        all_valid = True
        
        for var, allowed_values in self.PRODUCTION_VARS.items():
            value = os.getenv(var)
            
            if not value:
                self.warnings.append(f"{var} not set")
                all_valid = False
                continue
            
            if allowed_values and value not in allowed_values:
                self.warnings.append(
                    f"{var}='{value}' (valid: {', '.join(allowed_values)})"
                )
                all_valid = False
            else:
                self.info.append(f"✓ {var}={value}")
        
        return all_valid
    
    def check_database_connectivity(self) -> bool:
        """Test database connection (optional check)."""
        try:
            import psycopg2
            
            conn = psycopg2.connect(
                host=os.getenv("DB_HOST", "localhost"),
                port=int(os.getenv("DB_PORT", "5432")),
                database=os.getenv("POSTGRES_DB", "alpharl_quant"),
                user=os.getenv("POSTGRES_USER", "postgres"),
                password=os.getenv("POSTGRES_PASSWORD"),
                connect_timeout=3
            )
            conn.close()
            self.info.append("✓ Database connection successful")
            return True
        
        except ImportError:
            self.warnings.append("psycopg2 not installed - skipping DB check")
            return True
        
        except Exception as e:
            self.warnings.append(f"Database connection failed: {str(e)}")
            return False
    
    def validate_secrets_directory(self) -> bool:
        """Check .secrets directory structure."""
        secrets_dir = Path(".secrets")
        
        if not secrets_dir.exists():
            self.warnings.append(".secrets/ directory not found (optional)")
            return True
        
        # Check permissions
        stat_info = secrets_dir.stat()
        permissions = oct(stat_info.st_mode)[-3:]
        
        if permissions != "700":
            self.warnings.append(
                f".secrets/ has permissions {permissions} (should be 700)\n"
                f"  Fix with: chmod 700 .secrets"
            )
        
        self.info.append(f"✓ .secrets/ directory exists ({permissions})")
        return True
    
    def run_all_checks(self) -> bool:
        """Run all validation checks."""
        print(f"\n{BOLD}🔍 AlphaRL-Quant Secrets Validation{RESET}")
        print("=" * 60)
        print()
        
        # Load .env if it exists
        if Path(".env").exists():
            from dotenv import load_dotenv
            load_dotenv()
            print(f"{GREEN}✓{RESET} Loaded .env file")
        else:
            print(f"{RED}✗{RESET} .env file not found")
            return False
        
        print()
        
        # Run all checks
        checks = [
            ("File Existence", self.check_env_file_exists),
            ("Git Ignore", self.check_gitignore),
            ("Required Variables", self.validate_required_vars),
            ("Recommended Variables", self.validate_recommended_vars),
            ("Production Config", self.validate_production_vars),
            ("Secrets Directory", self.validate_secrets_directory),
            ("Database Connection", self.check_database_connectivity)
        ]
        
        results = {}
        for name, check_func in checks:
            print(f"Checking {name}...", end=" ")
            try:
                result = check_func()
                results[name] = result
                status = f"{GREEN}✓{RESET}" if result else f"{YELLOW}⚠{RESET}"
                print(status)
            except Exception as e:
                results[name] = False
                print(f"{RED}✗{RESET} {str(e)}")
        
        # Print summary
        print()
        print("=" * 60)
        print(f"{BOLD}Summary{RESET}")
        print("=" * 60)
        
        if self.errors:
            print(f"\n{RED}{BOLD}❌ ERRORS ({len(self.errors)}):{RESET}")
            for error in self.errors:
                print(f"  • {error}")
        
        if self.warnings:
            print(f"\n{YELLOW}{BOLD}⚠️  WARNINGS ({len(self.warnings)}):{RESET}")
            for warning in self.warnings:
                print(f"  • {warning}")
        
        if self.info:
            print(f"\n{GREEN}{BOLD}✅ SUCCESS ({len(self.info)}):{RESET}")
            for info in self.info:
                print(f"  • {info}")
        
        # Final verdict
        print()
        print("=" * 60)
        
        if self.errors:
            print(f"{RED}{BOLD}❌ VALIDATION FAILED{RESET}")
            print(f"Fix {len(self.errors)} error(s) before deploying to production")
            return False
        elif self.warnings:
            print(f"{YELLOW}{BOLD}⚠️  VALIDATION PASSED WITH WARNINGS{RESET}")
            print(f"Consider addressing {len(self.warnings)} warning(s)")
            return True
        else:
            print(f"{GREEN}{BOLD}✅ ALL VALIDATIONS PASSED{RESET}")
            print("Secrets are properly configured!")
            return True


def main():
    """Main entry point."""
    # Check if running from project root
    if not Path("scripts").exists():
        print(f"{RED}Error: Run this script from the project root directory{RESET}")
        sys.exit(1)
    
    # Install python-dotenv if not available
    try:
        import dotenv
    except ImportError:
        print(f"{YELLOW}Installing python-dotenv...{RESET}")
        os.system(f"{sys.executable} -m pip install -q python-dotenv")
    
    # Run validation
    validator = SecretsValidator()
    success = validator.run_all_checks()
    
    print()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
