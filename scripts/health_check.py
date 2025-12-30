"""
Health check script for pipeline.
Run before deployment or as monitoring check.
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.health import health_checker
from src.utils.logger import setup_logging, get_logger
import json

# Initialize logging
setup_logging()
logger = get_logger(__name__)


def main():
    """Run health checks and report status."""
    
    print("\n" + "=" * 60)
    print("Pipeline Health Check")
    print("=" * 60 + "\n")
    
    # Get health status
    status = health_checker.get_status()
    
    # Print status
    print(status)
    print()
    
    # Save to JSON
    status_file = Path("logs/health_status.json")
    status_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(status_file, 'w') as f:
        json.dump(status.to_dict(), f, indent=2)
    
    print(f"Status saved to: {status_file}")
    print()
    
    # Exit with appropriate code
    if status.healthy:
        print("✅ System is healthy and ready")
        sys.exit(0)
    else:
        print("❌ System has health issues - check logs")
        sys.exit(1)


if __name__ == "__main__":
    main()
