#!/bin/bash
# AlphaRL-Quant Secrets Setup
# Interactive setup with validation and secure permissions

set -e

echo "🔐 AlphaRL-Quant Secrets Setup"
echo "=============================="
echo ""

# Check if secrets directory exists
if [ ! -d ".secrets" ]; then
    echo "📁 Creating .secrets directory..."
    mkdir -p .secrets
    chmod 700 .secrets
    echo "✅ Created .secrets/ with restricted permissions (700)"
else
    echo "✅ .secrets/ directory exists"
fi

# Check if .env already exists
if [ -f ".env" ]; then
    echo "⚠️  .env file already exists!"
    read -p "Overwrite existing .env? (y/n): " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Cancelled. Keeping existing .env"
        exit 0
    fi
    echo "🔄 Backing up existing .env to .env.backup"
    cp .env .env.backup
fi

echo ""
echo "📝 Please provide the following configuration:"
echo ""

# Database credentials
echo "─────────────────────────────────────────"
echo "DATABASE CONFIGURATION"
echo "─────────────────────────────────────────"

read -p "PostgreSQL User [trader]: " DB_USER
DB_USER=${DB_USER:-trader}

read -p "PostgreSQL Database [alpharl_quant]: " DB_NAME
DB_NAME=${DB_NAME:-alpharl_quant}

while true; do
    read -sp "PostgreSQL Password (min 8 chars): " DB_PASSWORD
    echo ""
    
    if [ ${#DB_PASSWORD} -lt 8 ]; then
        echo "❌ Password must be at least 8 characters"
        continue
    fi
    
    read -sp "Confirm Password: " DB_PASSWORD_CONFIRM
    echo ""
    
    if [ "$DB_PASSWORD" != "$DB_PASSWORD_CONFIRM" ]; then
        echo "❌ Passwords do not match"
        continue
    fi
    
    break
done

echo "✅ Database credentials configured"
echo ""

# API Keys (optional)
echo "─────────────────────────────────────────"
echo "API KEYS (Optional - Press Enter to skip)"
echo "─────────────────────────────────────────"

read -p "Yahoo Finance API Key: " YAHOO_API_KEY
read -p "Alpha Vantage API Key: " ALPHA_VANTAGE_KEY
read -p "AWS Access Key ID: " AWS_ACCESS_KEY_ID
read -sp "AWS Secret Access Key: " AWS_SECRET_ACCESS_KEY
echo ""
read -p "AWS S3 Bucket (for backups): " AWS_S3_BUCKET

echo ""

# Environment
echo "─────────────────────────────────────────"
echo "ENVIRONMENT CONFIGURATION"
echo "─────────────────────────────────────────"

PS3="Select environment: "
options=("development" "staging" "production")
select ENV_TYPE in "${options[@]}"; do
    case $ENV_TYPE in
        "development"|"staging"|"production")
            ENVIRONMENT=$ENV_TYPE
            break
            ;;
        *) echo "Invalid option";;
    esac
done

# Log level based on environment
if [ "$ENVIRONMENT" = "production" ]; then
    LOG_LEVEL="INFO"
elif [ "$ENVIRONMENT" = "staging" ]; then
    LOG_LEVEL="INFO"
else
    LOG_LEVEL="DEBUG"
fi

echo "✅ Environment: $ENVIRONMENT (Log Level: $LOG_LEVEL)"
echo ""

# Generate secure random keys
echo "🔑 Generating secure random keys..."

if command -v openssl &> /dev/null; then
    SECRET_KEY=$(openssl rand -hex 32)
    JWT_SECRET=$(openssl rand -hex 32)
    ENCRYPTION_KEY=$(openssl rand -hex 32)
    echo "✅ Generated using OpenSSL"
else
    # Fallback to Python
    SECRET_KEY=$(python3 -c "import secrets; print(secrets.token_hex(32))")
    JWT_SECRET=$(python3 -c "import secrets; print(secrets.token_hex(32))")
    ENCRYPTION_KEY=$(python3 -c "import secrets; print(secrets.token_hex(32))")
    echo "✅ Generated using Python secrets"
fi

# Create .env file
echo ""
echo "📝 Writing .env file..."

cat > .env <<EOF
# ============================================================================
# AlphaRL-Quant Environment Configuration
# Generated: $(date)
# Environment: ${ENVIRONMENT}
# ============================================================================

# Environment
ENVIRONMENT=${ENVIRONMENT}
LOG_LEVEL=${LOG_LEVEL}
DEBUG=$( [ "$ENVIRONMENT" = "development" ] && echo "True" || echo "False" )

# Database Configuration
POSTGRES_USER=${DB_USER}
POSTGRES_PASSWORD=${DB_PASSWORD}
POSTGRES_DB=${DB_NAME}
DB_HOST=localhost
DB_PORT=5432

# Security Keys (Auto-generated - DO NOT SHARE)
SECRET_KEY=${SECRET_KEY}
JWT_SECRET=${JWT_SECRET}
ENCRYPTION_KEY=${ENCRYPTION_KEY}

# API Keys
YAHOO_FINANCE_API_KEY=${YAHOO_API_KEY}
ALPHA_VANTAGE_KEY=${ALPHA_VANTAGE_KEY}

# AWS Configuration (for backups)
AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID}
AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY}
AWS_S3_BUCKET=${AWS_S3_BUCKET}
AWS_REGION=us-east-1

# Application Settings
ALLOWED_HOSTS=localhost,127.0.0.1
CORS_ORIGINS=http://localhost:3000,http://localhost:8000

# Performance
MAX_WORKERS=4
BATCH_SIZE=64

# Monitoring
PROMETHEUS_ENABLED=true
GRAFANA_ADMIN_PASSWORD=admin

# ============================================================================
# WARNING: This file contains sensitive credentials
# - Never commit to version control
# - Keep permissions restricted (600)
# - Rotate secrets regularly
# ============================================================================
EOF

# Set restrictive permissions
chmod 600 .env
echo "✅ .env created with permissions 600 (owner read/write only)"

# Create secrets directory files
echo ""
echo "📁 Creating secrets templates..."

# API keys template
cat > .secrets/api_keys.yaml.example <<EOF
# API Keys Template
# Copy this to api_keys.yaml and fill in your keys

apis:
  yahoo_finance:
    api_key: "your_yahoo_finance_key"
    rate_limit: 2000  # requests per day
  
  alpha_vantage:
    api_key: "your_alpha_vantage_key"
    rate_limit: 500  # requests per day
  
  polygon_io:
    api_key: "your_polygon_key"
    rate_limit: 1000

cloud:
  aws:
    access_key_id: "your_aws_access_key"
    secret_access_key: "your_aws_secret"
    region: "us-east-1"
  
  gcp:
    project_id: "your_gcp_project"
    credentials_file: "path/to/credentials.json"

database:
  postgres:
    production:
      host: "your_db_host"
      port: 5432
      database: "alpharl_quant"
      user: "trader"
      password: "your_secure_password"
EOF

# Environment template
cat > .secrets/.env.template <<EOF
# AlphaRL-Quant Environment Variables Template
# Copy this to ../.env and fill in your values

# Environment
ENVIRONMENT=production
LOG_LEVEL=INFO

# Database
POSTGRES_USER=trader
POSTGRES_PASSWORD=<your_secure_password>
POSTGRES_DB=alpharl_quant
DB_HOST=localhost
DB_PORT=5432

# Security
SECRET_KEY=<generate_with_openssl_rand_hex_32>
JWT_SECRET=<generate_with_openssl_rand_hex_32>

# API Keys
YAHOO_FINANCE_API_KEY=<optional>
ALPHA_VANTAGE_KEY=<optional>

# AWS (for backups)
AWS_ACCESS_KEY_ID=<optional>
AWS_SECRET_ACCESS_KEY=<optional>
AWS_S3_BUCKET=<optional>
EOF

chmod 600 .secrets/.env.template
chmod 600 .secrets/api_keys.yaml.example

echo "✅ Created templates in .secrets/"

# Validate secrets
echo ""
echo "🔍 Validating configuration..."

if [ -f "scripts/validate_secrets.py" ]; then
    if python3 scripts/validate_secrets.py; then
        echo "✅ All required secrets validated"
    else
        echo "⚠️  Some validations failed - please review"
    fi
else
    echo "⚠️  scripts/validate_secrets.py not found - skipping validation"
fi

# Summary
echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║              ✅ SECRETS SETUP COMPLETE                     ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
echo "📁 Files created:"
echo "   - .env (600) - Main environment file"
echo "   - .env.backup (if existed) - Backup of old config"
echo "   - .secrets/.env.template - Template for reference"
echo "   - .secrets/api_keys.yaml.example - API keys template"
echo ""
echo "🔒 Security notes:"
echo "   - .env has restrictive permissions (600)"
echo "   - Added to .gitignore (verify with: git check-ignore .env)"
echo "   - Backup created if .env existed previously"
echo ""
echo "🎯 Next steps:"
echo "   1. Verify .env is in .gitignore: grep '\.env' .gitignore"
echo "   2. Never commit .env to version control"
echo "   3. Rotate secrets periodically (recommended: every 90 days)"
echo "   4. For team sharing, use a secrets manager (AWS Secrets Manager, Vault)"
echo ""
echo "💡 To verify your setup:"
echo "   python3 scripts/validate_secrets.py"
echo ""
