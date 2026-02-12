# AlphaRL-Quant Security Guide

## Overview

This document outlines the security practices, configurations, and best practices for deploying and operating AlphaRL-Quant in production environments.

**Security Principles**:
- 🔒 Defense in Depth
- 🔐 Least Privilege
- 🚫 Never Trust, Always Verify
- 📊 Continuous Monitoring
- 🔄 Regular Audits

---

## Table of Contents

- [Secrets Management](#secrets-management)
- [Authentication & Authorization](#authentication--authorization)
- [Docker Security](#docker-security)
- [Database Security](#database-security)
- [Network Security](#network-security)
- [Dependency Management](#dependency-management)
- [Monitoring & Auditing](#monitoring--auditing)
- [Incident Response](#incident-response)

---

## Secrets Management

### Setup

Use the interactive setup script:
```bash
bash scripts/setup_secrets.sh
```

This will:
- Create `.secrets/` directory with 700 permissions
- Generate secure random keys (64+ characters)
- Create `.env` with restrictive permissions (600)
- Validate password strength (min 8 chars)

### Validation

Verify your secrets configuration:
```bash
python3 scripts/validate_secrets.py
```

Expected output:
```
✅ ALL VALIDATIONS PASSED
Secrets are properly configured!
```

### Environment Variables

#### Required (Production)
```env
POSTGRES_PASSWORD=<strong_password>  # Min 8 chars, avoid common words
SECRET_KEY=<64_char_hex>             # For application encryption
JWT_SECRET=<64_char_hex>             # For JWT token signing
```

#### Optional (Recommended)
```env
YAHOO_FINANCE_API_KEY=<your_key>     # For higher rate limits
AWS_ACCESS_KEY_ID=<aws_key>          # For S3 backups
AWS_SECRET_ACCESS_KEY=<aws_secret>   
ENCRYPTION_KEY=<64_char_hex>         # For data-at-rest encryption
```

### Best Practices

1. **Never commit secrets to Git**
   ```bash
   # Verify .env is ignored
   git check-ignore .env
   # Should output: .env
   ```

2. **Rotate secrets regularly**
   ```bash
   # Recommended: Every 90 days
   # Re-run setup script
   bash scripts/setup_secrets.sh
   ```

3. **Use different secrets per environment**
   ```bash
   .env.development
   .env.staging
   .env.production
   ```

4. **For teams: Use a secrets manager**
   - AWS Secrets Manager
   - HashiCorp Vault
   - Azure Key Vault
   - Google Secret Manager

---

## Authentication & Authorization

### Database Access

**Principle of Least Privilege**:
```sql
-- Create read-only user for analytics
CREATE USER analyst WITH PASSWORD 'secure_password';
GRANT CONNECT ON DATABASE alpharl_quant TO analyst;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO analyst;

-- Create write-restricted user for pipeline
CREATE USER pipeline_user WITH PASSWORD 'secure_password';
GRANT CONNECT ON DATABASE alpharl_quant TO pipeline_user;
GRANT SELECT, INSERT, UPDATE ON market_data, pipeline_runs TO pipeline_user;
```

### API Authentication (Future)

When exposing APIs:
```python
# Use JWT tokens
from jose import jwt

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(hours=1)
    to_encode.update({"exp": expire})
    
    encoded_jwt = jwt.encode(
        to_encode,
        os.getenv("JWT_SECRET"),
        algorithm="HS256"
    )
    return encoded_jwt
```

---

## Docker Security

### Non-Root User

Our Dockerfile uses a non-root user:
```dockerfile
RUN useradd -m -u 1000 trader
USER trader
```

**Verification**:
```bash
docker-compose exec pipeline whoami
# Should output: trader
```

### Image Scanning

Scan for vulnerabilities before deployment:
```bash
# Install Trivy
brew install aquasecurity/trivy/trivy

# Scan image
trivy image alpharl-quant:latest

# Fail on HIGH/CRITICAL
trivy image --severity HIGH,CRITICAL --exit-code 1 alpharl-quant:latest
```

### Minimal Base Image

We use `python:3.10-slim` to minimize attack surface:
- 70% smaller than full Python image
- Fewer installed packages = fewer vulnerabilities

### Read-Only Filesystem (Advanced)

For extra hardening:
```yaml
# docker-compose.yml
services:
  pipeline:
    read_only: true
    tmpfs:
      - /tmp
      - /app/.cache
```

---

## Database Security

### Connection Security

**1. Use SSL/TLS for production**:
```python
# src/config.py
DATABASE_URL = f"postgresql://{user}:{password}@{host}:{port}/{db}?sslmode=require"
```

**2. Connection pooling limits**:
```yaml
# config/config.yaml
database:
  pool_size_min: 1
  pool_size_max: 10
  pool_timeout: 30
```

### Data Encryption

**At-rest encryption**:
```bash
# Enable PostgreSQL encryption
# postgresql.conf
ssl = on
ssl_cert_file = '/path/to/server.crt'
ssl_key_file = '/path/to/server.key'
```

**In-transit encryption**:
- Always use SSL connections
- Verify certificates: `sslmode=verify-full`

### Backup Encryption

```bash
# Encrypt backups before upload
gpg --symmetric --cipher-algo AES256 backup.sql
aws s3 cp backup.sql.gpg s3://bucket/backups/
```

---

## Network Security

### Firewall Rules

**Production deployment**:
```bash
# Allow only necessary ports
ufw allow 22/tcp    # SSH
ufw allow 80/tcp    # HTTP
ufw allow 443/tcp   # HTTPS
ufw allow 5432/tcp from 10.0.0.0/24  # PostgreSQL (internal only)
ufw enable
```

### Docker Network Isolation

```yaml
# docker-compose.yml
networks:
  frontend:
    driver: bridge
  backend:
    driver: bridge
    internal: true  # No internet access

services:
  postgres:
    networks:
      - backend  # Not exposed to internet
  
  api:
    networks:
      - frontend
      - backend
```

### CORS Configuration

```env
# .env
ALLOWED_HOSTS=yourdomain.com,api.yourdomain.com
CORS_ORIGINS=https://yourdomain.com,https://app.yourdomain.com
```

---

## Dependency Management

### Vulnerability Scanning

**1. Python packages**:
```bash
# Install safety
pip install safety

# Check for known vulnerabilities
safety check

# Check with detailed output
safety check --full-report
```

**2. Automated scanning (CI/CD)**:
```yaml
# .github/workflows/security.yml
- name: Security Check
  run: |
    pip install safety bandit
    safety check --exit-code 1
    bandit -r src/ -f json -o report.json
```

### Dependency Pinning

**requirements.txt** (pinned versions):
```txt
# Bad (unpinned)
pandas

# Good (pinned)
pandas==2.1.4

# Best (hash verification)
pandas==2.1.4 \
    --hash=sha256:abc123...
```

Generate with:
```bash
pip freeze > requirements.txt
```

### Regular Updates

```bash
# Check for outdated packages
pip list --outdated

# Update safely (test first!)
pip install --upgrade pandas
pytest tests/
```

---

## Monitoring & Auditing

### Logging Best Practices

**1. Structured JSON logging**:
```python
# src/utils/logging_config.py
import logging
import json

class SecurityLogger:
    @staticmethod
    def log_auth_attempt(user, success, ip):
        logging.info(json.dumps({
            "event": "auth_attempt",
            "user": user,
            "success": success,
            "ip": ip,
            "timestamp": datetime.utcnow().isoformat()
        }))
```

**2. Never log secrets**:
```python
# ❌ BAD
logger.info(f"Connecting with password: {password}")

# ✅ GOOD
logger.info("Connecting to database")
```

### Audit Trail

Track all critical operations:
```sql
-- Create audit table
CREATE TABLE audit_log (
    id SERIAL PRIMARY KEY,
    user_id INTEGER,
    action VARCHAR(50),
    resource VARCHAR(100),
    timestamp TIMESTAMP DEFAULT NOW(),
    ip_address INET,
    details JSONB
);

-- Index for queries
CREATE INDEX idx_audit_timestamp ON audit_log(timestamp DESC);
CREATE INDEX idx_audit_user ON audit_log(user_id);
```

### Security Metrics

Monitor via Prometheus:
```python
from prometheus_client import Counter

FAILED_AUTH_ATTEMPTS = Counter(
    'failed_auth_attempts_total',
    'Total failed authentication attempts'
)

SUSPICIOUS_ACTIVITY = Counter(
    'suspicious_activity_total',
    'Suspicious activity detected',
    ['type']
)
```

---

## Incident Response

### Preparation

**1. Have a response plan**:
- Incident detection
- Containment procedures
- Investigation steps
- Recovery process
- Post-mortem

**2. Emergency contacts**:
```yaml
# .secrets/emergency_contacts.yaml
security_team:
  - name: "Security Lead"
    email: "security@company.com"
    phone: "+1-XXX-XXX-XXXX"

cloud_provider:
  - aws_support: "https://console.aws.amazon.com/support"
  - gcp_support: "https://console.cloud.google.com/support"
```

### Detection

**Indicators of Compromise (IoC)**:
- Unusual database queries
- Failed authentication spikes
- Unexpected network traffic
- Suspicious file modifications

**Monitoring**:
```bash
# Check for suspicious processes
docker-compose exec pipeline ps aux

# Check network connections
docker-compose exec pipeline netstat -tuln

# Check file modifications
docker-compose exec pipeline find /app -mtime -1
```

### Containment

**Immediate actions**:
```bash
# 1. Isolate affected services
docker-compose stop affected_service

# 2. Rotate secrets immediately
bash scripts/setup_secrets.sh

# 3. Review logs
docker-compose logs --tail=1000 affected_service > incident_logs.txt

# 4. Backup current state
bash scripts/backup.sh

# 5. Notify security team
```

### Recovery

```bash
# 1. Restore from known-good backup
bash scripts/restore.sh backups/YYYYMMDD_HHMMSS.tar.gz

# 2. Verify integrity
python3 scripts/health_check.py

# 3. Update all dependencies
pip install --upgrade -r requirements.txt

# 4. Rebuild Docker images
docker-compose build --no-cache

# 5. Gradual rollout
docker-compose up -d postgres
docker-compose up -d pipeline
# Monitor logs continuously
```

---

## Security Checklist

### Pre-Deployment

- [ ] All secrets in `.env`, not in code
- [ ] `.env` has 600 permissions
- [ ] `.env` is in `.gitignore`
- [ ] Strong passwords (12+ chars, mixed case, special chars)
- [ ] Security keys are 64+ character random hex
- [ ] Secrets validation passes
- [ ] Dependency vulnerability scan clean
- [ ] Docker image vulnerability scan clean
- [ ] Database uses strong password
- [ ] Non-root user in containers
- [ ] SSL/TLS enabled for database
- [ ] Firewall configured
- [ ] Monitoring enabled
- [ ] Backup tested and working

### Post-Deployment

- [ ] Health checks passing
- [ ] Logs configured and monitored
- [ ] Alerts configured
- [ ] Incident response plan documented
- [ ] Team trained on security procedures
- [ ] Regular security reviews scheduled

### Ongoing

- [ ] Rotate secrets every 90 days
- [ ] Update dependencies monthly
- [ ] Scan for vulnerabilities weekly
- [ ] Review logs daily
- [ ] Backup verification monthly
- [ ] Security audit quarterly

---

## Tools & Resources

### Security Tools

```bash
# Vulnerability scanning
pip install safety bandit

# Secrets detection
pip install detect-secrets

# Docker security
docker run --rm -v /var/run/docker.sock:/var/run/docker.sock \
  aquasec/trivy:latest image alpharl-quant:latest
```

### External Resources

- [OWASP Top 10](https://owasp.org/www-project-top-ten/)
- [CIS Docker Benchmark](https://www.cisecurity.org/benchmark/docker)
- [Python Security Best Practices](https://python.readthedocs.io/en/stable/library/security_warnings.html)
- [PostgreSQL Security](https://www.postgresql.org/docs/current/security.html)

---

## Support

For security issues, DO NOT create public GitHub issues.

**Report security vulnerabilities**:
- Email: security@yourcompany.com
- PGP Key: [link to public key]
- Expected response: Within 24 hours

---

**Last Updated**: 2026-02-12  
**Next Review**: 2026-05-12 (90 days)

**Remember**: Security is not a one-time task, it's an ongoing process. Stay vigilant! 🔒
