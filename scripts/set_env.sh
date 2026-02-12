#!/bin/bash
# Environment Switching Utility for AlphaRL-Quant
# Safely switch between development, staging, and production environments

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# =============================================================================
# Configuration
# =============================================================================

VALID_ENVS=("development" "staging" "production")
CURRENT_ENV_FILE=".current_env"

# =============================================================================
# Functions
# =============================================================================

show_usage() {
    echo "Usage: $0 <environment>"
    echo ""
    echo "Environments:"
    echo "  development  - Local development with mock data"
    echo "  staging      - Pre-production testing"
    echo "  production   - Live production environment"
    echo ""
    echo "Example:"
    echo "  $0 development"
    echo "  $0 staging"
    echo "  $0 production"
}

validate_environment() {
    local env="$1"
    
    for valid_env in "${VALID_ENVS[@]}"; do
        if [ "$env" = "$valid_env" ]; then
            return 0
        fi
    done
    
    echo -e "${RED}Error: Invalid environment '$env'${NC}"
    echo "Valid environments: ${VALID_ENVS[*]}"
    return 1
}

get_current_env() {
    if [ -f "$CURRENT_ENV_FILE" ]; then
        cat "$CURRENT_ENV_FILE"
    else
        echo "none"
    fi
}

confirm_switch() {
    local from_env="$1"
    local to_env="$2"
    
    echo ""
    echo -e "${YELLOW}⚠️  Warning: Switching environments${NC}"
    echo "  From: $from_env"
    echo "  To:   $to_env"
    echo ""
    
    if [ "$to_env" = "production" ]; then
        echo -e "${RED}⚠️  PRODUCTION ENVIRONMENT${NC}"
        echo -e "${RED}This will use REAL data and REAL trading!${NC}"
        echo ""
    fi
    
    read -p "Continue? [y/N]: " -n 1 -r
    echo
    
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Cancelled"
        exit 0
    fi
}

validate_config() {
    local env="$1"
    
    echo -e "${BLUE}Validating $env configuration...${NC}"
    
    if ! python3 scripts/validate_env.py --env "$env"; then
        echo -e "${RED}❌ Configuration validation failed!${NC}"
        echo "Please fix the issues above before switching environments."
        return 1
    fi
    
    return 0
}

create_env_symlink() {
    local env="$1"
    
    # Create symlink to environment-specific config
    local env_config="config/environments/${env}.yaml"
    local active_config="config/active.yaml"
    
    if [ -L "$active_config" ]; then
        rm "$active_config"
    fi
    
    ln -s "environments/${env}.yaml" "$active_config"
    echo -e "${GREEN}✓${NC} Created symlink: config/active.yaml -> $env_config"
}

update_env_file() {
    local env="$1"
    
    # Update .env file with environment marker
    if [ -f ".env" ]; then
        # Remove old ENVIRONMENT line
        sed -i.bak '/^ENVIRONMENT=/d' .env
        rm .env.bak 2>/dev/null || true
    else
        touch .env
    fi
    
    # Add new ENVIRONMENT line
    echo "ENVIRONMENT=$env" >> .env
    echo -e "${GREEN}✓${NC} Updated .env file"
}

update_docker_compose() {
    local env="$1"
    
    # Set environment for docker-compose
    if [ -f "docker-compose.override.yml" ]; then
        rm docker-compose.override.yml
    fi
    
    if [ "$env" != "development" ]; then
        # Create override for non-dev environments
        cat > docker-compose.override.yml <<EOF
version: '3.8'

services:
  pipeline:
    environment:
      - ENVIRONMENT=$env
EOF
        echo -e "${GREEN}✓${NC} Created docker-compose.override.yml"
    fi
}

show_environment_info() {
    local env="$1"
    
    echo ""
    echo -e "${GREEN}╔══════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║  Environment: $env"
    echo -e "${GREEN}╚══════════════════════════════════════════════════════╝${NC}"
    echo ""
    
    case "$env" in
        development)
            echo "Features:"
            echo "  • Mock data enabled"
            echo "  • Debug mode active"
            echo "  • Relaxed validation"
            echo "  • Fast training (10k timesteps)"
            echo ""
            echo "Quick start:"
            echo "  docker-compose up -d"
            ;;
        staging)
            echo "Features:"
            echo "  • Real data, paper trading"
            echo "  • Strict validation"
            echo "  • Circuit breakers enabled"
            echo "  • Production-like settings"
            echo ""
            echo "Quick start:"
            echo "  bash scripts/deploy_docker.sh"
            ;;
        production)
            echo -e "${RED}⚠️  LIVE TRADING ENVIRONMENT${NC}"
            echo ""
            echo "Features:"
            echo "  • Real data, real trading"
            echo "  • Maximum safety features"
            echo "  • Comprehensive monitoring"
            echo "  • Automated backups"
            echo ""
            echo "Before trading:"
            echo "  1. Verify API keys are set"
            echo "  2. Run health check: python3 scripts/health_check.py"
            echo "  3. Check monitoring: open http://localhost:3000"
            echo "  4. Review trading limits in config"
            ;;
    esac
    
    echo ""
}

restart_services() {
    echo -e "${BLUE}Restarting services...${NC}"
    
    if docker-compose ps | grep -q "Up"; then
        docker-compose restart
        echo -e "${GREEN}✓${NC} Services restarted"
    else
        echo -e "${YELLOW}⚠${NC} No running services to restart"
    fi
}

# =============================================================================
# Main
# =============================================================================

main() {
    # Check arguments
    if [ $# -eq 0 ]; then
        show_usage
        exit 1
    fi
    
    local target_env="$1"
    
    # Validate environment
    if ! validate_environment "$target_env"; then
        exit 1
    fi
    
    # Get current environment
    local current_env=$(get_current_env)
    
    # Check if already in target environment
    if [ "$current_env" = "$target_env" ]; then
        echo -e "${YELLOW}Already in $target_env environment${NC}"
        show_environment_info "$target_env"
        exit 0
    fi
    
    # Confirm switch for production
    if [ "$target_env" = "production" ] || [ "$current_env" = "production" ]; then
        confirm_switch "$current_env" "$target_env"
    fi
    
    # Validate target environment configuration
    if ! validate_config "$target_env"; then
        exit 1
    fi
    
    echo ""
    echo -e "${BLUE}Switching to $target_env environment...${NC}"
    echo ""
    
    # Perform switch
    create_env_symlink "$target_env"
    update_env_file "$target_env"
    update_docker_compose "$target_env"
    
    # Save current environment
    echo "$target_env" > "$CURRENT_ENV_FILE"
    
    # Restart services if running
    if docker ps | grep -q "alpharl"; then
        read -p "Restart running services? [y/N]: " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            restart_services
        fi
    fi
    
    # Show environment info
    show_environment_info "$target_env"
    
    echo -e "${GREEN}✅ Successfully switched to $target_env environment${NC}"
    echo ""
}

# Run main
main "$@"
