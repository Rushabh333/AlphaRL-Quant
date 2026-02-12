# AWS Deployment Guide

This directory contains Terraform configuration for deploying AlphaRL-Quant to AWS.

## Architecture

### Infrastructure Components

- **VPC**: Multi-AZ networking with public and private subnets
- **ECS Fargate**: Serverless container hosting with auto-scaling
- **RDS PostgreSQL**: Managed database with multi-AZ for production
- **S3**: Model storage and backups with lifecycle policies
- **CloudWatch**: Comprehensive monitoring, dashboards, and alarms
- **Secrets Manager**: Secure credential storage

## Quick Start

### Prerequisites

```bash
# Install required tools
brew install awscli terraform

# Configure AWS credentials
aws configure
```

### Deploy to Staging

```bash
# 1. Set environment variables
export TF_VAR_db_password="your-secure-password"
export TF_VAR_alarm_email="team@example.com"

# 2. Run deployment script
bash scripts/deploy_aws.sh staging
```

### Deploy to Production

```bash
# 1. Set environment variables
export TF_VAR_db_password="your-very-secure-password"
export TF_VAR_ecr_repository_url="123456789012.dkr.ecr.us-east-1.amazonaws.com/alpharl-quant"
export TF_VAR_alarm_email="team@example.com"

# 2. Run deployment script (will prompt for confirmation)
bash scripts/deploy_aws.sh production
```

## Manual Terraform Deployment

```bash
cd terraform/aws

# Initialize
terraform init

# Plan (review changes)
terraform plan -var-file="staging.tfvars"

# Apply
terraform apply -var-file="staging.tfvars"

# View outputs
terraform output
```

## Infrastructure Features

### Auto-Scaling

- **CPU-based**: Scales when CPU > 75%
- **Memory-based**: Scales when memory > 80%
- **Min/Max**: 1-10 tasks (staging), 2-20 tasks (production)

### High Availability

- Multi-AZ deployment
- RDS with automatic failover (production)
- Read replica for disaster recovery (production)
- Cross-region S3 replication (production)

### Monitoring & Alerts

**CloudWatch Dashboard**: Real-time metrics for:
- ECS CPU & memory utilization
- RDS connections & storage
- Trading performance (Sharpe ratio)
- Error rates

**Alarms** (via SNS):
- High CPU/memory utilization
- Low database storage
- Tasks stopped unexpectedly
- High drawdown (> 10%)
- Model degradation (Sharpe ratio < 0.5)
- High API error rate (> 10%)

## Cost Optimization

### Staging Environment
- Spot instances (70% savings): `use_spot_instances = true`
- Smaller instances: `db.t3.medium`
- 7-day backup retention
- **Estimated cost**: $150-200/month

### Production Environment
- On-demand instances (reliability)
- Larger instances: `db.r6g.xlarge`
- 30-day backup retention
- Disaster recovery enabled
- **Estimated cost**: $600-800/month

## Security

All resources include:
- ✅ Encryption at rest (RDS, S3)
- ✅ Encryption in transit (SSL/TLS)
- ✅ Private subnets for database
- ✅ Security groups with least privilege
- ✅ Secrets Manager for credentials
- ✅ IAM roles with minimal permissions

## Disaster Recovery

Production includes:
- RDS read replica in alternate AZ
- Cross-region S3 replication to `us-west-2`
- Automated snapshots (30-day retention)
- **RTO**: ~15 minutes
- **RPO**: ~5 minutes

## Troubleshooting

### View ECS Logs
```bash
aws logs tail /ecs/alpharl-production --follow
```

### Check Service Status
```bash
aws ecs describe-services \
  --cluster alpharl-cluster-production \
  --services alpharl-pipeline-production
```

### Force New Deployment
```bash
aws ecs update-service \
  --cluster alpharl-cluster-production \
  --service alpharl-pipeline-production \
  --force-new-deployment
```

### Database Connection
```bash
# Get endpoint
terraform output db_endpoint

# Connect
psql -h <endpoint> -U trader -d alpharl_quant_production
```

## Cleanup

```bash
cd terraform/aws

# Destroy all resources
terraform destroy -var-file="staging.tfvars"
```

**⚠️ Warning**: This will delete all data including databases and backups!

## Additional Resources

- [AWS ECS Best Practices](https://docs.aws.amazon.com/AmazonECS/latest/bestpracticesguide/)
- [RDS Performance Insights](https://docs.aws.amazon.com/AmazonRDS/latest/UserGuide/USER_PerfInsights.html)
- [CloudWatch Container Insights](https://docs.aws.amazon.com/AmazonCloudWatch/latest/monitoring/ContainerInsights.html)
