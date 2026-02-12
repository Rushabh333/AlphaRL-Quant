#!/bin/bash
# Database initialization script for manual setup
# (For Docker, init_db.sql runs automatically)

set -e

echo "🗄️  Initializing AlphaRL-Quant Database..."
echo "=========================================="

# Check if PostgreSQL is running
if ! command -v psql &> /dev/null; then
    echo "❌ psql not found. Please install PostgreSQL first."
    exit 1
fi

# Load environment variables
if [ -f ".env" ]; then
    export $(cat .env | grep -v '^#' | xargs)
else
    echo "⚠️  No .env file found. Using default DB_PASSWORD."
fi

# Database connection parameters
DB_HOST="${DB_HOST:-localhost}"
DB_PORT="${DB_PORT:-5432}"
DB_NAME="${DB_NAME:-trading_db}"
DB_USER="${DB_USER:-postgres}"
DB_PASSWORD="${DB_PASSWORD:-changeme}"

# Test connection
echo "Testing connection to PostgreSQL..."
if ! PGPASSWORD=$DB_PASSWORD psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d postgres -c '\q' 2>/dev/null; then
    echo "❌ Cannot connect to PostgreSQL. Please check your connection settings."
    exit 1
fi
echo "✅ Connection successful"

# Create database if it doesn't exist
echo "Creating database '$DB_NAME' if not exists..."
PGPASSWORD=$DB_PASSWORD psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d postgres -tc "SELECT 1 FROM pg_database WHERE datname = '$DB_NAME'" | grep -q 1 || \
PGPASSWORD=$DB_PASSWORD psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d postgres -c "CREATE DATABASE $DB_NAME"

# Run initialization SQL
echo "Running schema initialization..."
PGPASSWORD=$DB_PASSWORD psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME -f scripts/init_db.sql

echo ""
echo "✅ Database initialized successfully!"
echo "Database: $DB_NAME"
echo "Host: $DB_HOST:$DB_PORT"
echo ""
echo "To connect: psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME"
