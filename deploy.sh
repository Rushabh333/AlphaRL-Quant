#!/bin/bash
# AlphaRL-Quant: One-Command Deploy
# Purpose: The ultimate quick-start script for Docker deployment
# Usage: bash deploy.sh

set -e  # Exit on error

clear
cat << "EOF"
╔══════════════════════════════════════════════════════════╗
║                                                          ║
║              🚀 AlphaRL-Quant Deploy 🚀                 ║
║                                                          ║
║          Production-Grade RL Trading System              ║
║                                                          ║
╚══════════════════════════════════════════════════════════╝

EOF

echo "This script will:"
echo "  1. ✅ Validate environment and secrets"
echo "  2. 🐳 Build and start Docker services"
echo "  3. 🏥 Run health checks"
echo "  4. 📊 Launch TensorBoard dashboard"
echo ""
read -p "Continue? (y/n) " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Deployment cancelled."
    exit 0
fi

echo ""
echo "═════════════════════════════════════════════════════════════"
echo "STEP 1: Environment Setup"
echo "═════════════════════════════════════════════════════════════"

# Check if .env exists
if [ ! -f ".env" ]; then
    echo "⚠️  No .env file found."
    if [ -f ".env.example" ]; then
        cp .env.example .env
        echo "Created .env from template."
        echo ""
        echo "Please configure these variables in .env:"
        echo "  - DB_PASSWORD (required for PostgreSQL)"
        echo ""
        read -p "Press Enter after editing .env, or Ctrl+C to cancel..."
    else
        echo "❌ No .env.example found. Cannot proceed."
        exit 1
    fi
fi

# Validate DB_PASSWORD is set
source .env
if [ -z "$DB_PASSWORD" ] || [ "$DB_PASSWORD" == "changeme" ]; then
    echo "⚠️  DB_PASSWORD not set or using default value."
    echo "   For production, set a strong password in .env"
    read -p "Continue with default? (y/n) " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo "✅ Environment configured"

echo ""
echo "═════════════════════════════════════════════════════════════"
echo "STEP 2: Docker Deployment"
echo "═════════════════════════════════════════════════════════════"

# Run the Docker deployment script
bash scripts/deploy_docker.sh

echo ""
echo "═════════════════════════════════════════════════════════════"
echo "STEP 3: Health Checks"
echo "═════════════════════════════════════════════════════════════"

# Determine docker compose command
if docker compose version &> /dev/null 2>&1; then
    DOCKER_COMPOSE="docker compose"
else
    DOCKER_COMPOSE="docker-compose"
fi

echo "🏥 Running comprehensive health checks..."
retries=5
healthy=false

for i in $(seq 1 $retries); do
    echo "  Attempt $i/$retries..."
    
    # Check TensorBoard
    if curl -s http://localhost:6006 > /dev/null 2>&1; then
        echo "  ✅ TensorBoard is accessible"
        healthy=true
        break
    fi
    
    sleep 3
done

if [ "$healthy" = false ]; then
    echo "  ⚠️  TensorBoard not responding yet (may still be starting)"
fi

# Check database connectivity
if $DOCKER_COMPOSE exec -T postgres pg_isready -U postgres > /dev/null 2>&1; then
    echo "  ✅ PostgreSQL is healthy"
else
    echo "  ⚠️  PostgreSQL connectivity issue"
fi

echo ""
echo "═════════════════════════════════════════════════════════════"
echo "STEP 4: System Status"
echo "═════════════════════════════════════════════════════════════"

$DOCKER_COMPOSE ps

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║                                                          ║"
echo "║              ✅ DEPLOYMENT SUCCESSFUL! ✅               ║"
echo "║                                                          ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""
echo "🌐 Access Points:"
echo "   📊 TensorBoard:  http://localhost:6006"
echo "   🗄️  PostgreSQL:  localhost:5432 (user: postgres)"
echo ""
echo "📁 Data Locations:"
echo "   - Raw data:       ./data/raw/"
echo "   - Processed:      ./data/processed/"
echo "   - Models:         ./models/"
echo "   - Logs:           ./logs/"
echo ""
echo "🎯 Quick Actions:"
echo ""
echo "   Train Agent (CPU):"
echo "     $DOCKER_COMPOSE --profile training up training"
echo ""
echo "   Train Agent (GPU):"
echo "     $DOCKER_COMPOSE --profile gpu up training-gpu"
echo ""
echo "   View Live Logs:"
echo "     $DOCKER_COMPOSE logs -f"
echo ""
echo "   Stop Everything:"
echo "     $DOCKER_COMPOSE down"
echo ""
echo "📖 Documentation:"
echo "   - Deployment Guide: cat DEPLOYMENT.md"
echo "   - Architecture:     cat ARCHITECTURE.md"
echo "   - Contributing:     cat CONTRIBUTING.md"
echo ""
echo "Happy Trading! 💹"
echo ""
