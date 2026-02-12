# Backup & Recovery Guide

## Overview

This guide covers the backup and recovery procedures for AlphaRL-Quant, including automated backups, cloud uploads, and disaster recovery.

**Critical Components**:
- 🗄️ PostgreSQL Database
- 🧠 Model Checkpoints
- ⚙️ Configuration Files
- 📝 Logs (optional)

---

## Table of Contents

- [Quick Start](#quick-start)
- [Automated Backups](#automated-backups)
- [Manual Backups](#manual-backups)
- [Cloud Storage](#cloud-storage)
- [Restore Procedures](#restore-procedures)
- [Disaster Recovery](#disaster-recovery)
- [Best Practices](#best-practices)

---

## Quick Start

### Create a Backup

```bash
# Basic backup (database + models + config)
bash scripts/backup.sh

# Full backup including logs
BACKUP_LOGS=true bash scripts/backup.sh

# Backup with cloud upload
UPLOAD_TO_S3=true AWS_S3_BUCKET=my-bucket bash scripts/backup.sh
```

### Restore from Backup

```bash
# List available backups
ls -lt backups/*.tar.gz | head -5

# Restore from specific backup
bash scripts/restore.sh backups/20260212_020000.tar.gz
```

---

## Automated Backups

### Setup Cron Jobs

**1. Install crontab**:
```bash
# Edit the crontab file
cp scripts/crontab.txt scripts/crontab_custom.txt

# Update paths in the file
sed -i 's|/path/to/AlphaRL-Quant|'$(pwd)'|g' scripts/crontab_custom.txt

# Review
cat scripts/crontab_custom.txt

# Install
crontab scripts/crontab_custom.txt

# Verify
crontab -l
```

**2. Recommended Schedule**:
- **Daily Backup**: 2:00 AM (database + models + config)
- **Weekly Full Backup**: Sunday 3:00 AM (includes logs)
- **Health Checks**: Every hour
- **Cleanup**: Monday 1:00 AM (remove old logs/backups)

### Configuration

Create `.backup.env`:
```bash
cp .backup.env.example .backup.env
nano .backup.env
```

**Key Settings**:
```bash
# Retention
BACKUP_RETENTION_DAYS=7

# Components
BACKUP_DATABASE=true
BACKUP_MODELS=true
BACKUP_CONFIG=true
BACKUP_LOGS=false  # Set to true for weekly backups

# Cloud upload
UPLOAD_TO_S3=true
AWS_S3_BUCKET=my-alpharl-backups
```

---

## Manual Backups

### Database Only

**Using Docker**:
```bash
docker-compose exec -T postgres pg_dump \
  -U trader alpharl_quant | gzip > backup_db.sql.gz
```

**Local PostgreSQL**:
```bash
PGPASSWORD=$POSTGRES_PASSWORD pg_dump \
  -h localhost -U trader alpharl_quant | gzip > backup_db.sql.gz
```

### Models Only

```bash
tar -czf backup_models.tar.gz models/
```

### Configuration Only

```bash
tar -czf backup_config.tar.gz config/ docker-compose.yml requirements.txt
```

### Full Manual Backup

```bash
# Create timestamped backup
timestamp=$(date +%Y%m%d_%H%M%S)
mkdir -p backups/$timestamp

# Database
docker-compose exec -T postgres pg_dump -U trader alpharl_quant \
  | gzip > backups/$timestamp/database.sql.gz

# Models
cp -r models backups/$timestamp/

# Config
cp -r config backups/$timestamp/
cp docker-compose.yml requirements.txt backups/$timestamp/

# Compress
tar -czf backups/${timestamp}.tar.gz -C backups $timestamp
rm -rf backups/$timestamp
```

---

## Cloud Storage

### AWS S3

**Prerequisites**:
```bash
# Install AWS CLI
pip install awscli

# Configure
aws configure
```

**Upload Backup**:
```bash
# Manual upload
aws s3 cp backups/20260212_020000.tar.gz \
  s3://my-bucket/alpharl-backups/

# Automated (in backup script)
UPLOAD_TO_S3=true AWS_S3_BUCKET=my-bucket bash scripts/backup.sh
```

**List Backups**:
```bash
aws s3 ls s3://my-bucket/alpharl-backups/
```

**Download Backup**:
```bash
aws s3 cp s3://my-bucket/alpharl-backups/20260212_020000.tar.gz ./backups/
```

### Google Cloud Storage

**Prerequisites**:
```bash
# Install gcloud CLI
# https://cloud.google.com/sdk/docs/install

# Authenticate
gcloud auth login
```

**Upload Backup**:
```bash
# Manual upload
gsutil cp backups/20260212_020000.tar.gz \
  gs://my-bucket/alpharl-backups/

# Automated
UPLOAD_TO_GCS=true GCP_BUCKET=my-bucket bash scripts/backup.sh
```

**List Backups**:
```bash
gsutil ls gs://my-bucket/alpharl-backups/
```

**Download Backup**:
```bash
gsutil cp gs://my-bucket/alpharl-backups/20260212_020000.tar.gz ./backups/
```

### Encryption (Optional)

For sensitive data:

```bash
# Encrypt before upload
gpg --symmetric --cipher-algo AES256 backups/20260212_020000.tar.gz

# Upload encrypted file
aws s3 cp backups/20260212_020000.tar.gz.gpg s3://my-bucket/

# Download and decrypt
aws s3 cp s3://my-bucket/20260212_020000.tar.gz.gpg ./
gpg --decrypt backups/20260212_020000.tar.gz.gpg > backup.tar.gz
```

---

## Restore Procedures

### Full Restore

```bash
# Interactive restore (recommended)
bash scripts/restore.sh backups/20260212_020000.tar.gz

# The script will:
# 1. Validate the backup archive
# 2. Show backup information
# 3. Prompt for each component
# 4. Create safety backups of existing data
# 5. Restore selected components
```

### Database Only

```bash
# Extract database from backup
tar -xzf backups/20260212_020000.tar.gz \
  20260212_020000/database/database.sql.gz

# Restore
gunzip -c 20260212_020000/database/database.sql.gz | \
  docker-compose exec -T postgres psql -U trader alpharl_quant
```

### Models Only

```bash
# Extract models
tar -xzf backups/20260212_020000.tar.gz \
  20260212_020000/models/

# Backup existing models
mv models models.backup.$(date +%Y%m%d_%H%M%S)

# Restore
cp -r 20260212_020000/models ./
```

### Verification After Restore

```bash
# 1. Check database
docker-compose exec postgres psql -U trader alpharl_quant -c "\dt"

# 2. Verify model files
ls -lh models/checkpoints/

# 3. Run health check
python3 scripts/health_check.py

# 4. Test pipeline
python3 scripts/run_pipeline.py --test
```

---

## Disaster Recovery

### Scenario 1: Database Corruption

```bash
# 1. Stop services
docker-compose down

# 2. Download latest backup from cloud
aws s3 cp s3://my-bucket/alpharl-backups/latest.tar.gz ./

# 3. Restore database
bash scripts/restore.sh latest.tar.gz
# Select: Database only

# 4. Restart services
docker-compose up -d

# 5. Verify
python3 scripts/health_check.py
```

### Scenario 2: Complete System Loss

```bash
# 1. Fresh clone/install
git clone <repo-url>
cd AlphaRL-Quant

# 2. Download backup
aws s3 cp s3://my-bucket/alpharl-backups/latest.tar.gz ./backups/

# 3. Restore everything
bash scripts/restore.sh backups/latest.tar.gz

# 4. Recreate secrets (not in backup for security)
bash scripts/setup_secrets.sh

# 5. Start services
docker-compose up -d

# 6. Verify
python3 scripts/health_check.py
```

### Scenario 3: Lost Model Checkpoints

```bash
# Download from cloud
aws s3 sync s3://my-bucket/alpharl-backups/models/ ./models/

# Or restore from latest backup
bash scripts/restore.sh backups/latest.tar.gz
# Select: Models only
```

### Recovery Time Objectives (RTO)

| Scenario | Recovery Time | Objective |
|----------|--------------|-----------|
| Database corruption | 5-10 minutes | < 15 minutes |
| Lost models | 2-5 minutes | < 10 minutes |
| Complete system loss | 15-30 minutes | < 1 hour |
| Config file corruption | 1-2 minutes | < 5 minutes |

---

## Best Practices

### Backup Strategy

**3-2-1 Rule**:
- **3** copies of data
- **2** different storage types
- **1** off-site backup

**Implementation**:
```bash
# Local backup (copy 1)
bash scripts/backup.sh

# Cloud backup S3 (copy 2, off-site)
UPLOAD_TO_S3=true bash scripts/backup.sh

# Cloud backup GCS (copy 3, off-site, different provider)
UPLOAD_TO_GCS=true bash scripts/backup.sh
```

### Retention Policy

**Recommended**:
- Daily backups: Keep 7 days
- Weekly backups: Keep 4 weeks
- Monthly backups: Keep 12 months

**Implementation**:
```bash
# Daily (7 days retention)
0 2 * * * BACKUP_RETENTION_DAYS=7 bash scripts/backup.sh

# Weekly (28 days retention)
0 3 * * 0 BACKUP_RETENTION_DAYS=28 BACKUP_LOGS=true bash scripts/backup.sh

# Monthly (365 days retention)
0 4 1 * * BACKUP_RETENTION_DAYS=365 BACKUP_LOGS=true bash scripts/backup.sh
```

### Testing Backups

**Monthly Test**:
```bash
# 1. Create test environment
mkdir -p test_restore
cd test_restore

# 2. Restore latest backup
bash ../scripts/restore.sh ../backups/latest.tar.gz

# 3. Verify database
cat database/database.sql.gz | gunzip | head -100

# 4. Verify models
ls -lh models/

# 5. Cleanup
cd ..
rm -rf test_restore
```

### Monitoring Backups

Add to `monitoring/rules/alerts.yml`:

```yaml
- alert: BackupFailed
  expr: time() - backup_last_success_timestamp > 86400
  for: 1h
  labels:
    severity: critical
  annotations:
    summary: "Backup has not run successfully in 24 hours"
```

### Security Considerations

1. **Never backup secrets in plain text**
   - `.env` file is sanitized in backups
   - Use secrets manager for production

2. **Encrypt sensitive backups**
   ```bash
   gpg --symmetric backups/latest.tar.gz
   ```

3. **Restrict backup access**
   ```bash
   chmod 600 backups/*.tar.gz
   chmod 700 backups/
   ```

4. **Use separate credentials for backups**
   - Dedicated S3 IAM user with minimal permissions
   - Separate GCS service account

---

## Troubleshooting

### Backup Script Fails

**Check logs**:
```bash
tail -50 logs/backup.log
```

**Common issues**:
```bash
# Docker not running
docker ps  # Should show containers

# Database not accessible
docker-compose exec postgres psql -U trader -c "SELECT 1"

# Disk space full
df -h

# Permissions
ls -la backups/
```

### Restore Script Fails

**Verify backup integrity**:
```bash
tar -tzf backups/20260212_020000.tar.gz | head
```

**Manual extraction**:
```bash
mkdir -p temp_restore
tar -xzf backups/20260212_020000.tar.gz -C temp_restore
ls -lR temp_restore/
```

### Cloud Upload Fails

**AWS S3**:
```bash
# Test credentials
aws s3 ls

# Test bucket access
aws s3 ls s3://my-bucket/

# Check IAM permissions
aws iam get-user
```

**GCS**:
```bash
# Test authentication
gcloud auth list

# Test bucket access
gsutil ls gs://my-bucket/

# Check permissions
gsutil iam get gs://my-bucket/
```

---

## Backup Checklist

### Daily
- [ ] Automated backup runs successfully
- [ ] Backup size is reasonable (not too small/large)
- [ ] Cloud upload succeeds (if configured)

### Weekly
- [ ] Full backup with logs completes
- [ ] Old backups cleaned up
- [ ] Backup space usage monitored

### Monthly
- [ ] Test restore procedure
- [ ] Verify backup integrity
- [ ] Review retention policy
- [ ] Update documentation

### Quarterly
- [ ] Disaster recovery drill
- [ ] Review and update backup strategy
- [ ] Audit backup security
- [ ] Test cross-region recovery

---

## Support

For backup-related issues:
1. Check logs: `tail -f logs/backup.log`
2. Run health check: `python3 scripts/health_check.py`
3. Verify disk space: `df -h`
4. Test backup manually: `bash scripts/backup.sh`

---

**Last Updated**: 2026-02-12  
**Next Review**: 2026-05-12
