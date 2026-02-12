#!/usr/bin/env python3
"""
Simple Pipeline Verification Script (no dependencies)

Checks AlphaRL-Quant pipeline completeness without requiring external packages.
"""

import os
import sys
from pathlib import Path

# Colors
GREEN = '\033[0;32m'
RED = '\033[0;31m'
YELLOW = '\033[1;33m'
BLUE = '\033[0;34m'
NC = '\033[0m'


def main():
    print(f"\n{BLUE}╔══════════════════════════════════════════════════════╗{NC}")
    print(f"{BLUE}║  AlphaRL-Quant Pipeline Verification                ║{NC}")
    print(f"{BLUE}╚══════════════════════════════════════════════════════╝{NC}\n")
    
    errors = []
    warnings = []
    passed = []
    
    # Check critical files and directories
    checks = {
        "Project Structure": [
            ('src', 'dir'),
            ('tests', 'dir'),
            ('scripts', 'dir'),
            ('config', 'dir'),
            ('docs', 'dir'),
            ('.github/workflows', 'dir'),
            ('terraform/aws', 'dir'),
        ],
        "Configuration Files": [
            ('config/base.yaml', 'file'),
            ('config/environments/development.yaml', 'file'),
            ('config/environments/staging.yaml', 'file'),
            ('config/environments/production.yaml', 'file'),
            ('.env.example', 'file'),
            ('docker-compose.yml', 'file'),
            ('Dockerfile', 'file'),
            ('requirements.txt', 'file'),
        ],
        "Scripts": [
            ('scripts/setup_secrets.sh', 'file'),
            ('scripts/validate_env.py', 'file'),
            ('scripts/set_env.sh', 'file'),
            ('scripts/backup.sh', 'file'),
            ('scripts/restore.sh', 'file'),
            ('scripts/deploy_aws.sh', 'file'),
            ('scripts/health_check.py', 'file'),
        ],
        "CI/CD Workflows": [
            ('.github/workflows/ci.yml', 'file'),
            ('.github/workflows/cd.yml', 'file'),
            ('.github/workflows/integration.yml', 'file'),
            ('.github/workflows/release.yml', 'file'),
        ],
        "Terraform Files": [
            ('terraform/aws/main.tf', 'file'),
            ('terraform/aws/variables.tf', 'file'),
            ('terraform/aws/outputs.tf', 'file'),
            ('terraform/aws/iam.tf', 'file'),
            ('terraform/aws/monitoring.tf', 'file'),
        ],
        "Documentation": [
            ('README.md', 'file'),
            ('docs/SECURITY.md', 'file'),
            ('docs/BACKUP_RECOVERY.md', 'file'),
            ('docs/COST_ANALYSIS.md', 'file'),
            ('terraform/aws/README.md', 'file'),
        ],
    }
    
    for category, items in checks.items():
        print(f"{BLUE}Checking {category}...{NC}")
        category_passed = 0
        category_failed = 0
        
        for path, item_type in items:
            full_path = Path(path)
            exists = full_path.exists()
            
            if item_type == 'dir':
                is_correct_type = full_path.is_dir()
            else:
                is_correct_type = full_path.is_file()
            
            if exists and is_correct_type:
                passed.append(path)
                category_passed += 1
            else:
                if category in ["Project Structure", "Configuration Files", "CI/CD Workflows"]:
                    errors.append(path)
                    category_failed += 1
                else:
                    warnings.append(path)
        
        if category_failed > 0:
            print(f"{RED}✗{NC} {category}: {category_passed} passed, {category_failed} failed\n")
        else:
            print(f"{GREEN}✓{NC} {category}: {category_passed}/{len(items)} present\n")
    
    # Summary
    print(f"{BLUE}╔══════════════════════════════════════════════════════╗{NC}")
    print(f"{BLUE}║  Verification Summary                                ║{NC}")
    print(f"{BLUE}╚══════════════════════════════════════════════════════╝{NC}\n")
    
    total_checks = len(passed) + len(errors) + len(warnings)
    print(f"{GREEN}✓ Passed: {len(passed)}/{total_checks}{NC}")
    print(f"{YELLOW}⚠ Warnings: {len(warnings)}{NC}")
    print(f"{RED}✗ Errors: {len(errors)}{NC}\n")
    
    if errors:
        print(f"{RED}Missing critical files/directories:{NC}")
        for error in errors[:10]:
            print(f"  {RED}•{NC} {error}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more")
        print()
    
    if warnings:
        print(f"{YELLOW}Missing optional files:{NC}")
        for warning in warnings[:5]:
            print(f"  {YELLOW}•{NC} {warning}")
        if len(warnings) > 5:
            print(f"  ... and {len(warnings) - 5} more")
        print()
    
    # Check file counts
    print(f"{BLUE}Quick Stats:{NC}")
    print(f"  Python files: {len(list(Path('src').rglob('*.py')))}")
    print(f"  Scripts: {len(list(Path('scripts').glob('*.py'))) + len(list(Path('scripts').glob('*.sh')))}")
    print(f"  Workflows: {len(list(Path('.github/workflows').glob('*.yml')))}")
    print(f"  Terraform files: {len(list(Path('terraform/aws').glob('*.tf')))}")
    print()
    
    if not errors:
        print(f"{GREEN}🎉 Pipeline verification complete!{NC}")
        print(f"{GREEN}All critical components are present and ready.{NC}\n")
        return 0
    else:
        print(f"{RED}❌ Pipeline has missing critical components.{NC}\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
