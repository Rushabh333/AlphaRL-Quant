# Environment: staging

environment = "staging"
aws_region  = "us-east-1"

# Availability zones
availability_zones = ["us-east-1a", "us-east-1b", "us-east-1c"]

# Database configuration
db_instance_class       = "db.t3.medium"
db_allocated_storage    = 50
db_max_allocated_storage = 200
db_name                 = "alpharl_quant_staging"
db_username             = "trader"
db_backup_retention_days = 7

# IMPORTANT: Set via environment variable
# export TF_VAR_db_password="your-secure-password"

# ECS configuration
pipeline_cpu           = 1024  # 1 vCPU
pipeline_memory        = 2048  # 2 GB
pipeline_desired_count = 2
training_cpu           = 4096  # 4 vCPU
training_memory        = 8192  # 8 GB
use_spot_instances     = true  # 70% cost savings

# Auto-scaling
autoscaling_min_capacity = 1
autoscaling_max_capacity = 5

# IMPORTANT: Set via environment variable
# export TF_VAR_ecr_repository_url="123456789012.dkr.ecr.us-east-1.amazonaws.com/alpharl-quant"

image_tag = "staging-latest"

# Storage
backup_retention_days = 30

# Disaster recovery (disabled for staging)
enable_disaster_recovery = false

# Monitoring
enable_detailed_monitoring = true

# OPTIONAL: Set via environment variable for email alerts
# export TF_VAR_alarm_email="team@example.com"
