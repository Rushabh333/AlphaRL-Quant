#!/bin/bash
# Docker deployment script for AlphaRL-Quant
# Author: AlphaRL-Quant Team
# Purpose: Deploy the entire system using Docker Compose

set -e  # Exit on error

echo "🐳 AlphaRL-Quant Docker Deployment"
echo "==================================="
echo ""

# =============================================================================
# 1. Check Docker Installation
# =============================================================================
echo "📋 Checking prerequisites..."

if ! command -v docker &> /dev/null; then
    echo"❌ Docker not found. Please install Docker Desktop."
    echo "   Download from: https://www.docker.com/products/docker-desktop"
    exit 1
fi

if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker Desktop."
    exit 1
fi

echo "✅ Docker is running"

# Check docker-compose
if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null 2>&1; then
    echo "❌ docker-compose not found. Please install docker-compose."
    exit 1
fi

# Use 'docker compose' (v2) if available, otherwise 'docker-compose' (v1)
if docker compose version &> /dev/null 2>&1; then
    DOCKER_COMPOSE="docker compose"
else
    DOCKER_COMPOSE="docker-compose"
fi

echo "✅ Docker Compose is available"

# =============================================================================
# 2. Environment Setup
# =============================================================================
echo "🔧 Setting up environment..."

if [ ! -f ".env" ]; then
    echo "⚠️  No .env file found."
    if [ -f ".env.example" ]; then
        cp .env.example .env
        echo "✅ Created .env from .env.example"
        echo "⚠️  Please edit .env with your credentials and run this script again"
        exit 1
    else
        echo "❌ .env.example not found. Cannot proceed."
        exit 1
    fi
fi

# Load environment variables
export $(cat .env | grep -v '^#' | xargs)
echo "✅ Environment variables loaded"

# =============================================================================
# 3. Build Docker Images
# =============================================================================
echo ""
echo "🔨 Building Docker images..."
echo "This may take 5-10 minutes on first run..."
echo ""

$DOCKER_COMPOSE build pipeline tensorboard

echo "✅ Images built successfully"

# =============================================================================
# 4. Start Core Services
# =============================================================================
echo ""
echo "🚀 Starting core services..."

$DOCKER_COMPOSE up -d postgres tensorboard

echo "✅ PostgreSQL and TensorBoard starting..."

# =============================================================================
# 5. Wait for PostgreSQL
# =============================================================================
echo "⏳ Waiting for PostgreSQL to be ready..."

retries=30
until $DOCKER_COMPOSE exec -T postgres pg_isready -U postgres > /dev/null 2>&1 || [ $retries -eq 0 ]; do
    retries=$((retries-1))
    echo -n "."
    sleep 2
done

if [ $retries -eq 0 ]; then
    echo ""
    echo "❌ PostgreSQL failed to start. Check logs with: $DOCKER_COMPOSE logs postgres"
    exit 1
fi

echo ""
echo "✅ PostgreSQL is ready"

# =============================================================================
# 6. Initialize Database
# =============================================================================
echo "💾 Initializing database schema..."

if $DOCKER_COMPOSE exec -T postgres psql -U postgres -d trading_db -f /docker-entrypoint-initdb.d/init.sql > /dev/null 2>&1; then
    echo "✅ Database initialized"
else
    echo "⚠️  Database may already be initialized (non-critical)"
fi

# =============================================================================
# 7. Run Data Pipeline
# =============================================================================
echo ""
echo "🔄 Running data pipeline..."
echo "This will fetch data and engineer features..."
echo ""

if $DOCKER_COMPOSE up pipeline; then
    echo "✅ Pipeline completed successfully"
else
    echo "❌ Pipeline failed. Check logs with: $DOCKER_COMPOSE logs pipeline"
    exit 1
fi

# =============================================================================
# 8. Services Status
# =============================================================================
echo ""
echo "📊 Services Status:"
$DOCKER_COMPOSE ps

# =============================================================================
# 9. Success Summary
# =============================================================================
echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║            ✅ DOCKER DEPLOYMENT COMPLETE                   ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
echo "🌐 Services:"
echo "   📊 TensorBoard: http://localhost:6006"
echo "   🗄️  PostgreSQL: localhost:5432"
echo ""
echo "🎯 Next Steps:"
echo ""
echo "   1. Train Model (CPU):"
echo "      $DOCKER_COMPOSE --profile training up training"
echo ""
echo "   2. Train Model (GPU, if available):"
echo "      $DOCKER_COMPOSE --profile gpu up training-gpu"
echo ""
echo "   3. View Logs:"
echo "      $DOCKER_COMPOSE logs -f [service-name]"
echo ""
echo "   4. Stop All Services:"
echo "      $DOCKER_COMPOSE down"
echo ""
echo "   5. Stop and Remove Volumes:"
echo "      $DOCKER_COMPOSE down -v"
echo ""
echo "💡 Useful Commands:"
echo "   - Shell into container: $DOCKER_COMPOSE exec pipeline bash"
echo "   - View all services: $DOCKER_COMPOSE ps"
echo "   - Check resource usage: docker stats"
echo ""
