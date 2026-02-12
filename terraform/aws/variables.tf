# AWS Infrastructure Variables

# =============================================================================
# General Configuration
# =============================================================================

variable "aws_region" {
  description = "AWS region for infrastructure"
  type        = string
  default     = "us-east-1"
}

variable "environment" {
  description = "Environment name (development, staging, production)"
  type        = string
  validation {
    condition     = contains(["development", "staging", "production"], var.environment)
    error_message = "Environment must be development, staging, or production."
  }
}

variable "availability_zones" {
  description = "List of availability zones"
  type        = list(string)
  default     = ["us-east-1a", "us-east-1b", "us-east-1c"]
}

# =============================================================================
# Networking
# =============================================================================

variable "vpc_cidr" {
  description = "CIDR block for VPC"
  type        = string
  default     = "10.0.0.0/16"
}

# =============================================================================
# Database (RDS)
# =============================================================================

variable "db_instance_class" {
  description = "RDS instance class"
  type        = string
  default     = "db.t3.small"
}

variable "db_allocated_storage" {
  description = "Initial database storage in GB"
  type        = number
  default     = 20
}

variable "db_max_allocated_storage" {
  description = "Maximum database storage for autoscaling in GB"
  type        = number
  default     = 100
}

variable "db_name" {
  description = "Database name"
  type        = string
  default     = "alpharl_quant"
}

variable "db_username" {
  description = "Database master username"
  type        = string
  default     = "trader"
}

variable "db_password" {
  description = "Database master password"
  type        = string
  sensitive   = true
}

variable "db_backup_retention_days" {
  description = "Number of days to retain database backups"
  type        = number
  default     = 7
}

# =============================================================================
# ECS Configuration
# =============================================================================

variable "pipeline_cpu" {
  description = "CPU units for pipeline task (256, 512, 1024, 2048, 4096)"
  type        = number
  default     = 512
}

variable "pipeline_memory" {
  description = "Memory for pipeline task in MB"
  type        = number
  default     = 1024
}

variable "pipeline_desired_count" {
  description = "Desired number of pipeline tasks"
  type        = number
  default     = 1
}

variable "training_cpu" {
  description = "CPU units for training task"
  type        = number
  default     = 4096
}

variable "training_memory" {
  description = "Memory for training task in MB"
  type        = number
  default     = 8192
}

variable "use_spot_instances" {
  description = "Use Fargate Spot instances (70% cost savings)"
  type        = bool
  default     = true
}

# =============================================================================
# Auto-Scaling
# =============================================================================

variable "autoscaling_min_capacity" {
  description = "Minimum number of ECS tasks"
  type        = number
  default     = 1
}

variable "autoscaling_max_capacity" {
  description = "Maximum number of ECS tasks"
  type        = number
  default     = 10
}

# =============================================================================
# Container Registry
# =============================================================================

variable "ecr_repository_url" {
  description = "URL of ECR repository for Docker images"
  type        = string
}

variable "image_tag" {
  description = "Docker image tag to deploy"
  type        = string
  default     = "latest"
}

# =============================================================================
# Storage (S3)
# =============================================================================

variable "backup_retention_days" {
  description = "Number of days to retain backups in S3"
  type        = number
  default     = 90
}

# =============================================================================
# Disaster Recovery
# =============================================================================

variable "enable_disaster_recovery" {
  description = "Enable multi-region disaster recovery"
  type        = bool
  default     = false
}

variable "dr_region" {
  description = "Disaster recovery region"
  type        = string
  default     = "us-west-2"
}

# =============================================================================
# Monitoring
# =============================================================================

variable "enable_detailed_monitoring" {
  description = "Enable detailed CloudWatch monitoring"
  type        = bool
  default     = true
}

variable "alarm_email" {
  description = "Email address for CloudWatch alarms"
  type        = string
  default     = ""
}

# =============================================================================
# Tags
# =============================================================================

variable "additional_tags" {
  description = "Additional tags to apply to all resources"
  type        = map(string)
  default     = {}
}
