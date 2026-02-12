# Environment: production

environment = "production"
aws_region  = "us-east-1"

# Availability zones
availability_zones = ["us-east-1a", "us-east-1b", "us-east-1c"]

# Database configuration (Production-grade)
db_instance_class        = "db.r6g.xlarge"  # 4 vCPU, 32 GB RAM
db_allocated_storage     = 100
db_max_allocated_storage = 500
db_name                  = "alpharl_quant_production"
db_username              = "trader"
db_backup_retention_days = 30

# IMPORTANT: Set via AWS Secrets Manager or environment variable
# export TF_VAR_db_password="your-very-secure-password"

# ECS configuration (Production-grade)
pipeline_cpu           = 2048  # 2 vCPU
pipeline_memory        = 4096  # 4 GB
pipeline_desired_count = 3     # High availability
training_cpu           = 8192  # 8 vCPU
training_memory        = 16384 # 16 GB
use_spot_instances     = false # Use on-demand for reliability

# Auto-scaling
autoscaling_min_capacity = 2
autoscaling_max_capacity = 20

# IMPORTANT: Set via environment variable
# export TF_VAR_ecr_repository_url="123456789012.dkr.ecr.us-east-1.amazonaws.com/alpharl-quant"

image_tag = "production-latest"

# Storage
backup_retention_days = 90

# Disaster recovery (ENABLED for production)
enable_disaster_recovery = true
dr_region                = "us-west-2"

# Monitoring
enable_detailed_monitoring = true

# IMPORTANT: Set via environment variable for critical alerts
# export TF_VAR_alarm_email="team@example.com"

# Additional production tags
additional_tags = {
  CostCenter  = "Trading"
  Compliance  = "SOC2"
  DataClass   = "Confidential"
}
