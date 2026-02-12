# GCP Deployment Quick Start

This directory contains configurations for deploying AlphaRL-Quant to Google Cloud Platform.

## Architecture (Simpler than AWS)

- **Cloud Run**: Serverless containers with auto-scaling
- **Cloud SQL**: Managed PostgreSQL database
- **Cloud Storage**: Model storage and backups
- **Cloud Monitoring**: Dashboards and alerts

## Quick Deploy

### Prerequisites

```bash
# Install gcloud CLI
brew install --cask google-cloud-sdk

# Authenticate
gcloud auth login
gcloud config set project YOUR_PROJECT_ID
```

### Deploy Script

```bash
#!/bin/bash
# Quick GCP deployment

PROJECT_ID="your-project-id"
REGION="us-central1"

# Enable APIs
gcloud services enable run.googleapis.com sqladmin.googleapis.com

# Create Cloud SQL instance
gcloud sql instances create alpharl-db \
  --database-version=POSTGRES_15 \
  --cpu=2 \
  --memory=4GB \
  --region=$REGION

# Create database
gcloud sql databases create alpharl_quant --instance=alpharl-db

# Build and deploy to Cloud Run
gcloud builds submit --tag gcr.io/$PROJECT_ID/alpharl-quant
gcloud run deploy alpharl-pipeline \
  --image gcr.io/$PROJECT_ID/alpharl-quant \
  --platform managed \
  --region $REGION \
  --allow-unauthenticated \
  --memory 4Gi \
  --cpu 2 \
  --min-instances 1 \
  --max-instances 10

# Create storage bucket
gsutil mb -l $REGION gs://$PROJECT_ID-alpharl-models

echo "✅ Deployment complete!"
```

## Cost Comparison

### GCP vs AWS

| Component | AWS (staging) | GCP (staging) |
|-----------|---------------|---------------|
| Compute   | ECS Fargate: $50/mo | Cloud Run: $30/mo |
| Database  | RDS: $80/mo | Cloud SQL: $60/mo |
| Storage   | S3: $20/mo | GCS: $20/mo |
| **Total** | **~$150/mo** | **~$110/mo** |

**Savings**: ~25% cheaper on GCP for simple workloads

## Why Use GCP?

✅ Simpler setup (< 10 commands vs Terraform)  
✅ Pay-per-use pricing (Cloud Run scales to zero)  
✅ Integrated with AI Platform for model serving  
✅ Better for experimentation and development  

## Why Use AWS?

✅ More granular control  
✅ Better enterprise features  
✅ More third-party integrations  
✅ Preferred if already using AWS ecosystem  

## Monitoring

```bash
# View logs
gcloud run services logs read alpharl-pipeline

# Create monitoring dashboard
# (Use GCP Console: Monitoring > Dashboards)
```

## Full Terraform (Optional)

For production GCP deployments, consider creating Terraform configs similar to AWS. The current quick-start is sufficient for staging/development.

## Cleanup

```bash
gcloud run services delete alpharl-pipeline
gcloud sql instances delete alpharl-db
gsutil rm -r gs://$PROJECT_ID-alpharl-models
```
