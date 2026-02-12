---
description: Deploy AlphaRL-Quant system  
---

# Deployment Workflow

Choose your deployment method:

## Quick Start

```bash
# One-command Docker deploy (recommended)
bash deploy.sh
```

## Deployment Options

### 1. Local Development Deployment
```bash
bash scripts/deploy_local.sh
```
- Creates virtual environment
- Installs dependencies  
- Runs data pipeline
- Best for: Development, testing, debugging

### 2. Docker Deployment
```bash
bash scripts/deploy_docker.sh
```
- Builds Docker images
- Starts all services (PostgreSQL, TensorBoard)
- Runs pipeline in container
- Best for: Production, consistency, portability

### 3. Manual Pipeline Run
```bash
# After deployment
python scripts/run_pipeline.py
```

### 4. Train RL Model

**CPU Training:**
```bash
# Local
python src/training/train_agent.py

# Docker
docker compose --profile training up training
```

**GPU Training (Docker only):**
```bash
docker compose --profile gpu up training-gpu
```

### 5. Monitor Training
```bash
# Start TensorBoard (local)
tensorboard --logdir=./logs/tensorboard/

# Visit http://localhost:6006

# Docker: TensorBoard starts automatically
# Access at http://localhost:6006
```

### 6. Run Backtest
```bash
python src/evaluation/backtest.py
```

## Service Management (Docker)

```bash
# View running services
docker compose ps

# View logs
docker compose logs -f [service-name]

# Stop all services
docker compose down

# Stop and remove data
docker compose down -v

# Restart service
docker compose restart [service-name]
```

## Troubleshooting

**Issue: Docker fails to start**
- Ensure Docker Desktop is running
- Check disk space: `df -h`

**Issue: Python version error**
- Install Python 3.10+: `brew install python@3.10`

**Issue: Database connection error**
- Check DB_PASSWORD in .env
- Verify PostgreSQL is running: `docker compose ps postgres`

For more details, see [DEPLOYMENT.md](../DEPLOYMENT.md)
