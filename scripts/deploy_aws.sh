#!/bin/bash
# AWS Deployment Script for AlphaRL-Quant
# Automated deployment to AWS using Terraform and ECS

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

ENVIRONMENT="${1:-staging}"
AWS_REGION="${AWS_REGION:-us-east-1}"
TERRAFORM_DIR="terraform/aws"
ECR_REPOSITORY_NAME="alpharl-quant"

echo -e "${BLUE}╔══════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║  AlphaRL-Quant AWS Deployment                       ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════╝${NC}"
echo ""
echo "Environment: ${ENVIRONMENT}"
echo "AWS Region: ${AWS_REGION}"
echo ""

# =============================================================================
# Pre-flight Checks
# =============================================================================

echo -e "${BLUE}Running pre-flight checks...${NC}"

# Check AWS CLI
if ! command -v aws &> /dev/null; then
    echo -e "${RED}❌ AWS CLI not found. Please install it first.${NC}"
    exit 1
fi

# Check Terraform
if ! command -v terraform &> /dev/null; then
    echo -e "${RED}❌ Terraform not found. Please install it first.${NC}"
    exit 1
fi

# Check Docker
if ! command -v docker &> /dev/null; then
    echo -e "${RED}❌ Docker not found. Please install it first.${NC}"
    exit 1
fi

# Verify AWS credentials
if ! aws sts get-caller-identity &> /dev/null; then
    echo -e "${RED}❌ AWS credentials not configured.${NC}"
    echo "Run: aws configure"
    exit 1
fi

AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
echo -e "${GREEN}✓${NC} AWS Account: $AWS_ACCOUNT_ID"

# Verify environment configuration
if ! python3 scripts/validate_env.py --env "$ENVIRONMENT"; then
    echo -e "${RED}❌ Environment validation failed!${NC}"
    exit 1
fi
echo -e "${GREEN}✓${NC} Environment configuration valid"

echo ""

# =============================================================================
# Create ECR Repository (if not exists)
# =============================================================================

echo -e "${BLUE}Setting up ECR repository...${NC}"

ECR_REPO_URI=$(aws ecr describe-repositories \
    --repository-names "$ECR_REPOSITORY_NAME" \
    --region "$AWS_REGION" \
    --query 'repositories[0].repositoryUri' \
    --output text 2>/dev/null || echo "")

if [ -z "$ECR_REPO_URI" ]; then
    echo "Creating ECR repository..."
    ECR_REPO_URI=$(aws ecr create-repository \
        --repository-name "$ECR_REPOSITORY_NAME" \
        --image-scanning-configuration scanOnPush=true \
        --encryption-configuration encryptionType=AES256 \
        --region "$AWS_REGION" \
        --query 'repository.repositoryUri' \
        --output text)
    echo -e "${GREEN}✓${NC} Created ECR repository: $ECR_REPO_URI"
else
    echo -e "${GREEN}✓${NC} ECR repository exists: $ECR_REPO_URI"
fi

echo ""

# =============================================================================
# Build and Push Docker Image
# =============================================================================

echo -e "${BLUE}Building Docker image...${NC}"

# Build both production and training images
docker build --target production -t alpharl-quant:latest .
docker build --target training -t alpharl-quant:latest-training .

# Tag images
IMAGE_TAG="${ENVIRONMENT}-$(date +%Y%m%d-%H%M%S)"
docker tag alpharl-quant:latest "${ECR_REPO_URI}:${IMAGE_TAG}"
docker tag alpharl-quant:latest "${ECR_REPO_URI}:${ENVIRONMENT}-latest"
docker tag alpharl-quant:latest-training "${ECR_REPO_URI}:${IMAGE_TAG}-training"

echo -e "${GREEN}✓${NC} Built images with tag: $IMAGE_TAG"

# Login to ECR
echo "Logging in to ECR..."
aws ecr get-login-password --region "$AWS_REGION" | \
    docker login --username AWS --password-stdin "$ECR_REPO_URI"

# Push images
echo "Pushing images to ECR..."
docker push "${ECR_REPO_URI}:${IMAGE_TAG}"
docker push "${ECR_REPO_URI}:${ENVIRONMENT}-latest"
docker push "${ECR_REPO_URI}:${IMAGE_TAG}-training"

echo -e "${GREEN}✓${NC} Pushed images to ECR"
echo ""

# =============================================================================
# Deploy Infrastructure with Terraform
# =============================================================================

echo -e "${BLUE}Deploying infrastructure with Terraform...${NC}"

cd "$TERRAFORM_DIR"

# Initialize Terraform
terraform init

# Create terraform.tfvars if not exists
if [ ! -f "terraform.tfvars" ]; then
    cat > terraform.tfvars <<EOF
environment = "$ENVIRONMENT"
aws_region  = "$AWS_REGION"
ecr_repository_url = "$ECR_REPO_URI"
image_tag   = "$IMAGE_TAG"

# Database credentials (use environment variables)
db_password = "$DB_PASSWORD"

# Alarm email
alarm_email = "${ALARM_EMAIL:-}"

# Scaling configuration
autoscaling_min_capacity = 1
autoscaling_max_capacity = ${AUTOSCALING_MAX:-10}

# Enable DR for production
enable_disaster_recovery = ${ENABLE_DR:-false}
EOF
    echo -e "${GREEN}✓${NC} Created terraform.tfvars"
fi

# Plan
echo "Planning infrastructure changes..."
terraform plan -out=tfplan

# Apply with confirmation
if [ "$ENVIRONMENT" = "production" ]; then
    echo -e "${YELLOW}⚠️  PRODUCTION DEPLOYMENT${NC}"
    read -p "Continue with production deployment? [y/N]: " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Deployment cancelled"
        exit 0
    fi
fi

terraform apply tfplan

echo -e "${GREEN}✓${NC} Infrastructure deployed"

# Save outputs
terraform output -json > outputs.json
echo -e "${GREEN}✓${NC} Saved Terraform outputs"

cd ../..
echo ""

# =============================================================================
# Configure Application
# =============================================================================

echo -e "${BLUE}Configuring application...${NC}"

# Extract database endpoint
DB_ENDPOINT=$(cd "$TERRAFORM_DIR" && terraform output -raw db_address)
ECS_CLUSTER=$(cd "$TERRAFORM_DIR" && terraform output -raw ecs_cluster_name)
ECS_SERVICE=$(cd "$TERRAFORM_DIR" && terraform output -raw pipeline_service_name)

echo "Database: $DB_ENDPOINT"
echo "ECS Cluster: $ECS_CLUSTER"
echo "ECS Service: $ECS_SERVICE"
echo ""

# =============================================================================
# Database Initialization
# =============================================================================

echo -e "${BLUE}Initializing database...${NC}"

# Run database initialization task
aws ecs run-task \
    --cluster "$ECS_CLUSTER" \
    --task-definition "alpharl-pipeline-${ENVIRONMENT}" \
    --launch-type FARGATE \
    --network-configuration "awsvpcConfiguration={subnets=[subnet-xxx],securityGroups=[sg-xxx]}" \
    --overrides '{
        "containerOverrides": [{
            "name": "pipeline",
            "command": ["python", "scripts/init_db.py"]
        }]
    }' \
    --region "$AWS_REGION" > /dev/null 2>&1 || echo "DB init task failed (may already be initialized)"

echo -e "${GREEN}✓${NC} Database initialization triggered"
echo ""

# =============================================================================
# Health Check
# =============================================================================

echo -e "${BLUE}Running health checks...${NC}"

sleep 30  # Wait for service to stabilize

# Check ECS service status
SERVICE_STATUS=$(aws ecs describe-services \
    --cluster "$ECS_CLUSTER" \
    --services "$ECS_SERVICE" \
    --region "$AWS_REGION" \
    --query 'services[0].status' \
    --output text)

if [ "$SERVICE_STATUS" = "ACTIVE" ]; then
    echo -e "${GREEN}✓${NC} ECS service is active"
else
    echo -e "${YELLOW}⚠${NC} ECS service status: $SERVICE_STATUS"
fi

# Check task count
RUNNING_TASKS=$(aws ecs describe-services \
    --cluster "$ECS_CLUSTER" \
    --services "$ECS_SERVICE" \
    --region "$AWS_REGION" \
    --query 'services[0].runningCount' \
    --output text)

echo "Running tasks: $RUNNING_TASKS"
echo ""

# =============================================================================
# Summary
# =============================================================================

echo -e "${GREEN}╔══════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║  ✅ Deployment Complete!                             ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════════════════╝${NC}"
echo ""
echo "Environment: $ENVIRONMENT"
echo "Image: ${ECR_REPO_URI}:${IMAGE_TAG}"
echo "Database: $DB_ENDPOINT"
echo ""
echo "Next steps:"
echo "  • View logs: aws logs tail /ecs/alpharl-${ENVIRONMENT} --follow"
echo "  • CloudWatch Dashboard: https://console.aws.amazon.com/cloudwatch/home?region=${AWS_REGION}#dashboards:name=AlphaRL-${ENVIRONMENT}"
echo "  • ECS Console: https://console.aws.amazon.com/ecs/home?region=${AWS_REGION}#/clusters/${ECS_CLUSTER}/services/${ECS_SERVICE}"
echo ""
echo -e "${BLUE}Deployment completed successfully!${NC}"
