#!/bin/bash
# AlphaRL-Quant Restore Script
# Restores database, models, and configuration from backup archive

set -e  # Exit on error
set -u  # Exit on undefined variable

# =============================================================================
# Configuration
# =============================================================================

# Load environment variables
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# =============================================================================
# Utility Functions
# =============================================================================

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

confirm() {
    local prompt="$1"
    local default="${2:-n}"
    
    if [ "$default" = "y" ]; then
        prompt="$prompt [Y/n]: "
    else
        prompt="$prompt [y/N]: "
    fi
    
    read -p "$prompt" -n 1 -r
    echo
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        return 0
    elif [[ $REPLY =~ ^[Nn]$ ]]; then
        return 1
    else
        # Use default
        [[ $default == "y" ]]
    fi
}

# =============================================================================
# Restore Functions
# =============================================================================

validate_backup() {
    local archive_file="$1"
    
    log_info "Validating backup archive..."
    
    # Check if file exists
    if [ ! -f "$archive_file" ]; then
        log_error "Backup file not found: $archive_file"
        return 1
    fi
    
    # Check if it's a valid tar.gz
    if ! tar -tzf "$archive_file" &>/dev/null; then
        log_error "Invalid or corrupted backup archive"
        return 1
    fi
    
    # Check for manifest
    if ! tar -tzf "$archive_file" | grep -q "manifest.json"; then
        log_warning "No manifest found in backup (old format?)"
    fi
    
    log_success "Backup archive is valid"
    return 0
}

extract_backup() {
    local archive_file="$1"
    local restore_dir="$2"
    
    log_info "Extracting backup archive..."
    
    mkdir -p "$restore_dir"
    tar -xzf "$archive_file" -C "$restore_dir" --strip-components=1
    
    log_success "Backup extracted to $restore_dir"
}

show_backup_info() {
    local restore_dir="$1"
    
    if [ -f "$restore_dir/manifest.json" ]; then
        log_info "Backup Information:"
        echo ""
        
        # Parse and display manifest (requires jq or basic parsing)
        if command -v jq &> /dev/null; then
            cat "$restore_dir/manifest.json" | jq '.'
        else
            cat "$restore_dir/manifest.json"
        fi
        
        echo ""
    fi
}

restore_database() {
    local restore_dir="$1"
    local db_backup="$restore_dir/database/database.sql.gz"
    
    if [ ! -f "$db_backup" ]; then
        log_warning "Database backup not found in archive"
        return 1
    fi
    
    log_warning "This will REPLACE the current database!"
    
    if ! confirm "Restore database?"; then
        log_info "Skipping database restore"
        return 0
    fi
    
    log_info "Restoring PostgreSQL database..."
    
    # Decompress
    gunzip -c "$db_backup" > "/tmp/alpharl_restore.sql"
    
    # Check if we're using Docker or local PostgreSQL
    if docker-compose ps postgres | grep -q "Up"; then
        log_info "Using Docker PostgreSQL..."
        
        # Restore to database
        cat "/tmp/alpharl_restore.sql" | docker-compose exec -T postgres psql \
            -U "${POSTGRES_USER:-postgres}" \
            -d "${POSTGRES_DB:-alpharl_quant}"
    elif command -v psql &> /dev/null; then
        log_info "Using local PostgreSQL..."
        
        PGPASSWORD="${POSTGRES_PASSWORD}" psql \
            -h "${DB_HOST:-localhost}" \
            -p "${DB_PORT:-5432}" \
            -U "${POSTGRES_USER:-postgres}" \
            -d "${POSTGRES_DB:-alpharl_quant}" \
            -f "/tmp/alpharl_restore.sql"
    else
        log_error "PostgreSQL not accessible"
        rm -f "/tmp/alpharl_restore.sql"
        return 1
    fi
    
    # Cleanup
    rm -f "/tmp/alpharl_restore.sql"
    
    log_success "Database restored"
    return 0
}

restore_models() {
    local restore_dir="$1"
    local models_backup="$restore_dir/models"
    
    if [ ! -d "$models_backup" ]; then
        log_warning "Models backup not found in archive"
        return 1
    fi
    
    log_warning "This will REPLACE existing model files!"
    
    if ! confirm "Restore models?"; then
        log_info "Skipping models restore"
        return 0
    fi
    
    log_info "Restoring model checkpoints..."
    
    # Backup existing models (just in case)
    if [ -d "models" ] && [ -n "$(ls -A models 2>/dev/null)" ]; then
        local backup_timestamp=$(date +%Y%m%d_%H%M%S)
        mv models "models.backup.$backup_timestamp"
        log_info "Existing models moved to models.backup.$backup_timestamp"
    fi
    
    # Restore models
    mkdir -p models
    cp -r "$models_backup"/* models/
    
    local model_count=$(find models -type f | wc -l | tr -d ' ')
    log_success "Restored $model_count model files"
    
    return 0
}

restore_configuration() {
    local restore_dir="$1"
    local config_backup="$restore_dir/config"
    
    if [ ! -d "$config_backup" ]; then
        log_warning "Configuration backup not found in archive"
        return 1
    fi
    
    log_info "Restoring configuration files..."
    
    # Restore YAML configs
    if [ -d "$config_backup/config" ]; then
        if confirm "Restore config/*.yaml files?"; then
            cp -r "$config_backup/config" ./
            log_success "Configuration files restored"
        fi
    fi
    
    # Restore docker-compose files
    for file in docker-compose.yml docker-compose.monitoring.yml; do
        if [ -f "$config_backup/$file" ]; then
            if confirm "Restore $file?"; then
                cp "$config_backup/$file" ./
                log_success "$file restored"
            fi
        fi
    done
    
    # Note about env file
    if [ -f "$config_backup/env.template" ]; then
        log_info "Environment template available at: $config_backup/env.template"
        log_warning "Review and manually merge with your .env (contains no secrets)"
    fi
    
    return 0
}

restore_logs() {
    local restore_dir="$1"
    local logs_backup="$restore_dir/logs.tar.gz"
    
    if [ ! -f "$logs_backup" ]; then
        log_info "No logs in backup (expected)"
        return 0
    fi
    
    if confirm "Restore logs?"; then
        log_info "Restoring logs..."
        
        mkdir -p logs/restored
        tar -xzf "$logs_backup" -C logs/restored
        
        log_success "Logs restored to logs/restored/"
    fi
    
    return 0
}

# =============================================================================
# Main Execution
# =============================================================================

main() {
    echo ""
    echo "╔════════════════════════════════════════════════════════════╗"
    echo "║         AlphaRL-Quant Restore Utility                     ║"
    echo "╚════════════════════════════════════════════════════════════╝"
    echo ""
    
    # Check arguments
    if [ $# -eq 0 ]; then
        log_error "Usage: $0 <backup_file.tar.gz>"
        echo ""
        log_info "Example: $0 backups/20260212_010000.tar.gz"
        echo ""
        
        # List available backups
        if [ -d "backups" ] && [ -n "$(ls -A backups/*.tar.gz 2>/dev/null)" ]; then
            log_info "Available backups:"
            ls -1t backups/*.tar.gz | head -5
        fi
        
        exit 1
    fi
    
    local archive_file="$1"
    local restore_dir="./backups/restore_temp"
    
    log_info "Starting restore from: $archive_file"
    echo ""
    
    # Validate backup
    if ! validate_backup "$archive_file"; then
        exit 1
    fi
    
    # Extract backup
    extract_backup "$archive_file" "$restore_dir"
    
    # Show backup info
    show_backup_info "$restore_dir"
    
    # Confirm restore
    echo ""
    log_warning "⚠️  IMPORTANT: This will replace existing data!"
    echo ""
    
    if ! confirm "Continue with restore?" "n"; then
        log_info "Restore cancelled"
        rm -rf "$restore_dir"
        exit 0
    fi
    
    echo ""
    
    # Restore components
    local components_restored=0
    
    restore_database "$restore_dir" && ((components_restored++)) || true
    restore_models "$restore_dir" && ((components_restored++)) || true
    restore_configuration "$restore_dir" && ((components_restored++)) || true
    restore_logs "$restore_dir" || true
    
    # Cleanup
    log_info "Cleaning up temporary files..."
    rm -rf "$restore_dir"
    
    # Summary
    echo ""
    echo "╔════════════════════════════════════════════════════════════╗"
    echo "║                 RESTORE COMPLETE                           ║"
    echo "╚════════════════════════════════════════════════════════════╝"
    echo ""
    
    if [ $components_restored -gt 0 ]; then
        log_success "$components_restored component(s) restored successfully"
        echo ""
        log_info "Next steps:"
        echo "  1. Verify database: docker-compose exec postgres psql -U trader -d alpharl_quant"
        echo "  2. Check models: ls -lh models/"
        echo "  3. Review config: cat config/config.yaml"
        echo "  4. Run health check: python3 scripts/health_check.py"
        echo ""
    else
        log_warning "No components were restored"
    fi
}

# Run main function
main "$@"
