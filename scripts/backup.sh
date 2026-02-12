#!/bin/bash
# AlphaRL-Quant Automated Backup Script
# Backs up database, models, configuration, and optionally uploads to cloud

set -e  # Exit on error
set -u  # Exit on undefined variable

# =============================================================================
# Configuration
# =============================================================================

# Load environment variables
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

# Backup configuration
BACKUP_ROOT="${BACKUP_ROOT:-./backups}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="$BACKUP_ROOT/$TIMESTAMP"
RETENTION_DAYS="${BACKUP_RETENTION_DAYS:-7}"

# Components to backup
BACKUP_DATABASE="${BACKUP_DATABASE:-true}"
BACKUP_MODELS="${BACKUP_MODELS:-true}"
BACKUP_CONFIG="${BACKUP_CONFIG:-true}"
BACKUP_LOGS="${BACKUP_LOGS:-false}"

# Cloud upload (optional)
UPLOAD_TO_S3="${UPLOAD_TO_S3:-false}"
UPLOAD_TO_GCS="${UPLOAD_TO_GCS:-false}"
AWS_S3_BUCKET="${AWS_S3_BUCKET:-}"
GCP_BUCKET="${GCP_BUCKET:-}"

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

check_command() {
    if ! command -v "$1" &> /dev/null; then
        log_error "$1 is not installed"
        return 1
    fi
    return 0
}

get_size_human() {
    if [ -f "$1" ]; then
        du -sh "$1" | cut -f1
    elif [ -d "$1" ]; then
        du -sh "$1" | cut -f1
    else
        echo "0B"
    fi
}

# =============================================================================
# Backup Functions
# =============================================================================

backup_database() {
    log_info "Backing up PostgreSQL database..."
    
    local db_backup="$BACKUP_DIR/database"
    mkdir -p "$db_backup"
    
    # Check if we're using Docker or local PostgreSQL
    if docker-compose ps postgres | grep -q "Up"; then
        # Docker-based backup
        log_info "Using Docker PostgreSQL..."
        docker-compose exec -T postgres pg_dump \
            -U "${POSTGRES_USER:-postgres}" \
            -d "${POSTGRES_DB:-alpharl_quant}" \
            --clean --if-exists \
            > "$db_backup/database.sql"
        
        # Also backup schema only (for quick reference)
        docker-compose exec -T postgres pg_dump \
            -U "${POSTGRES_USER:-postgres}" \
            -d "${POSTGRES_DB:-alpharl_quant}" \
            --schema-only \
            > "$db_backup/schema.sql"
    elif command -v pg_dump &> /dev/null; then
        # Local PostgreSQL backup
        log_info "Using local PostgreSQL..."
        PGPASSWORD="${POSTGRES_PASSWORD}" pg_dump \
            -h "${DB_HOST:-localhost}" \
            -p "${DB_PORT:-5432}" \
            -U "${POSTGRES_USER:-postgres}" \
            -d "${POSTGRES_DB:-alpharl_quant}" \
            --clean --if-exists \
            > "$db_backup/database.sql"
    else
        log_warning "PostgreSQL not accessible, skipping database backup"
        return 1
    fi
    
    # Compress database backup
    gzip -f "$db_backup/database.sql"
    
    local size=$(get_size_human "$db_backup/database.sql.gz")
    log_success "Database backed up ($size)"
    
    return 0
}

backup_models() {
    log_info "Backing up model checkpoints..."
    
    local models_backup="$BACKUP_DIR/models"
    mkdir -p "$models_backup"
    
    local model_count=0
    
    # Backup model checkpoints
    if [ -d "models/checkpoints" ] && [ -n "$(ls -A models/checkpoints 2>/dev/null)" ]; then
        cp -r models/checkpoints "$models_backup/"
        model_count=$(find models/checkpoints -type f | wc -l | tr -d ' ')
        log_info "Copied $model_count checkpoint files"
    fi
    
    # Backup best models
    if [ -d "models/best" ] && [ -n "$(ls -A models/best 2>/dev/null)" ]; then
        cp -r models/best "$models_backup/"
    fi
    
    # Backup final models
    if [ -f "models/ppo_trading_final.zip" ]; then
        cp models/ppo_trading_final.zip "$models_backup/"
    fi
    
    if [ $model_count -eq 0 ]; then
        log_warning "No model checkpoints found"
        return 1
    fi
    
    local size=$(get_size_human "$models_backup")
    log_success "Models backed up ($size, $model_count files)"
    
    return 0
}

backup_configuration() {
    log_info "Backing up configuration files..."
    
    local config_backup="$BACKUP_DIR/config"
    mkdir -p "$config_backup"
    
    # Backup YAML configs
    if [ -d "config" ]; then
        cp -r config "$config_backup/"
    fi
    
    # Backup environment file (sanitized)
    if [ -f ".env" ]; then
        # Create sanitized copy (remove sensitive values)
        grep -v "PASSWORD\|SECRET\|KEY\|TOKEN" .env > "$config_backup/env.template" || true
        log_info "Created sanitized env template"
    fi
    
    # Backup docker-compose files
    [ -f "docker-compose.yml" ] && cp docker-compose.yml "$config_backup/"
    [ -f "docker-compose.monitoring.yml" ] && cp docker-compose.monitoring.yml "$config_backup/"
    
    # Backup requirements
    [ -f "requirements.txt" ] && cp requirements.txt "$config_backup/"
    
    # Backup important documentation
    for doc in README.md DEPLOYMENT.md SECURITY.md; do
        [ -f "$doc" ] && cp "$doc" "$config_backup/"
    done
    
    local size=$(get_size_human "$config_backup")
    log_success "Configuration backed up ($size)"
    
    return 0
}

backup_logs() {
    log_info "Backing up logs..."
    
    local logs_backup="$BACKUP_DIR/logs"
    mkdir -p "$logs_backup"
    
    # Backup recent logs (last 7 days)
    if [ -d "logs" ]; then
        find logs -name "*.log" -mtime -7 -exec cp {} "$logs_backup/" \;
    fi
    
    # Compress logs
    if [ -n "$(ls -A $logs_backup 2>/dev/null)" ]; then
        tar -czf "$logs_backup.tar.gz" -C "$BACKUP_DIR" logs/
        rm -rf "$logs_backup"
        
        local size=$(get_size_human "$logs_backup.tar.gz")
        log_success "Logs backed up ($size)"
        return 0
    else
        log_warning "No recent logs found"
        rmdir "$logs_backup" 2>/dev/null || true
        return 1
    fi
}

create_manifest() {
    log_info "Creating backup manifest..."
    
    cat > "$BACKUP_DIR/manifest.json" <<EOF
{
  "backup_timestamp": "$TIMESTAMP",
  "backup_date": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "hostname": "$(hostname)",
  "components": {
    "database": $( [ -f "$BACKUP_DIR/database/database.sql.gz" ] && echo "true" || echo "false" ),
    "models": $( [ -d "$BACKUP_DIR/models" ] && echo "true" || echo "false" ),
    "config": $( [ -d "$BACKUP_DIR/config" ] && echo "true" || echo "false" ),
    "logs": $( [ -f "$BACKUP_DIR/logs.tar.gz" ] && echo "true" || echo "false" )
  },
  "environment": "${ENVIRONMENT:-development}",
  "version": "1.0"
}
EOF
    
    log_success "Manifest created"
}

compress_backup() {
    log_info "Compressing backup archive..."
    
    local archive_name="$BACKUP_ROOT/${TIMESTAMP}.tar.gz"
    
    tar -czf "$archive_name" -C "$BACKUP_ROOT" "$TIMESTAMP"
    
    # Remove uncompressed directory
    rm -rf "$BACKUP_DIR"
    
    local size=$(get_size_human "$archive_name")
    log_success "Backup compressed: $archive_name ($size)"
    
    echo "$archive_name"
}

upload_to_s3() {
    local archive_file="$1"
    
    if [ "$UPLOAD_TO_S3" != "true" ] || [ -z "$AWS_S3_BUCKET" ]; then
        return 0
    fi
    
    log_info "Uploading to S3..."
    
    if ! check_command "aws"; then
        log_error "AWS CLI not installed, skipping S3 upload"
        return 1
    fi
    
    local s3_path="s3://$AWS_S3_BUCKET/alpharl-backups/$(basename $archive_file)"
    
    if aws s3 cp "$archive_file" "$s3_path" --no-progress; then
        log_success "Uploaded to S3: $s3_path"
        return 0
    else
        log_error "S3 upload failed"
        return 1
    fi
}

upload_to_gcs() {
    local archive_file="$1"
    
    if [ "$UPLOAD_TO_GCS" != "true" ] || [ -z "$GCP_BUCKET" ]; then
        return 0
    fi
    
    log_info "Uploading to Google Cloud Storage..."
    
    if ! check_command "gsutil"; then
        log_error "gsutil not installed, skipping GCS upload"
        return 1
    fi
    
    local gcs_path="gs://$GCP_BUCKET/alpharl-backups/$(basename $archive_file)"
    
    if gsutil cp "$archive_file" "$gcs_path"; then
        log_success "Uploaded to GCS: $gcs_path"
        return 0
    else
        log_error "GCS upload failed"
        return 1
    fi
}

cleanup_old_backups() {
    log_info "Cleaning up old backups (keeping last $RETENTION_DAYS days)..."
    
    local deleted_count=0
    
    # Find and delete old backups
    while IFS= read -r old_backup; do
        rm -f "$old_backup"
        ((deleted_count++))
    done < <(find "$BACKUP_ROOT" -name "*.tar.gz" -mtime +$RETENTION_DAYS)
    
    if [ $deleted_count -gt 0 ]; then
        log_success "Deleted $deleted_count old backup(s)"
    else
        log_info "No old backups to delete"
    fi
}

# =============================================================================
# Main Execution
# =============================================================================

main() {
    echo ""
    echo "╔════════════════════════════════════════════════════════════╗"
    echo "║         AlphaRL-Quant Backup Utility                      ║"
    echo "╚════════════════════════════════════════════════════════════╝"
    echo ""
    
    log_info "Starting backup at $(date)"
    log_info "Backup directory: $BACKUP_DIR"
    
    # Create backup directory
    mkdir -p "$BACKUP_DIR"
    
    # Track what was backed up
    local components_backed_up=0
    
    # Execute backups
    if [ "$BACKUP_DATABASE" = "true" ]; then
        backup_database && ((components_backed_up++)) || true
    fi
    
    if [ "$BACKUP_MODELS" = "true" ]; then
        backup_models && ((components_backed_up++)) || true
    fi
    
    if [ "$BACKUP_CONFIG" = "true" ]; then
        backup_configuration && ((components_backed_up++)) || true
    fi
    
    if [ "$BACKUP_LOGS" = "true" ]; then
        backup_logs && ((components_backed_up++)) || true
    fi
    
    # Check if anything was backed up
    if [ $components_backed_up -eq 0 ]; then
        log_error "No components were successfully backed up"
        rm -rf "$BACKUP_DIR"
        exit 1
    fi
    
    # Create manifest
    create_manifest
    
    # Compress backup
    archive_file=$(compress_backup)
    
    # Upload to cloud (if configured)
    upload_to_s3 "$archive_file" || true
    upload_to_gcs "$archive_file" || true
    
    # Cleanup old backups
    cleanup_old_backups
    
    # Summary
    echo ""
    echo "╔════════════════════════════════════════════════════════════╗"
    echo "║                 BACKUP COMPLETE                            ║"
    echo "╚════════════════════════════════════════════════════════════╝"
    echo ""
    log_success "Archive: $archive_file"
    log_success "Size: $(get_size_human $archive_file)"
    log_success "Components: $components_backed_up backed up"
    echo ""
    log_info "To restore: bash scripts/restore.sh $archive_file"
    echo ""
}

# Run main function
main "$@"
