# AWS Infrastructure Outputs

# =============================================================================
# VPC Outputs
# =============================================================================

output "vpc_id" {
  description = "ID of the VPC"
  value       = aws_vpc.main.id
}

output "public_subnet_ids" {
  description = "IDs of public subnets"
  value       = aws_subnet.public[*].id
}

output "private_subnet_ids" {
  description = "IDs of private subnets"
  value       = aws_subnet.private[*].id
}

# =============================================================================
# Database Outputs
# =============================================================================

output "db_endpoint" {
  description = "RDS PostgreSQL endpoint"
  value       = aws_db_instance.postgres.endpoint
  sensitive   = true
}

output "db_address" {
  description = "RDS PostgreSQL address"
  value       = aws_db_instance.postgres.address
}

output "db_port" {
  description = "RDS PostgreSQL port"
  value       = aws_db_instance.postgres.port
}

output "db_name" {
  description = "Database name"
  value       = aws_db_instance.postgres.db_name
}

# =============================================================================
# ECS Outputs
# =============================================================================

output "ecs_cluster_name" {
  description = "Name of ECS cluster"
  value       = aws_ecs_cluster.main.name
}

output "ecs_cluster_arn" {
  description = "ARN of ECS cluster"
  value       = aws_ecs_cluster.main.arn
}

output "pipeline_service_name" {
  description = "Name of pipeline ECS service"
  value       = aws_ecs_service.pipeline.name
}

# =============================================================================
# S3 Outputs
# =============================================================================

output "models_bucket" {
  description = "S3 bucket for model storage"
  value       = aws_s3_bucket.models.id
}

output "backups_bucket" {
  description = "S3 bucket for backups"
  value       = aws_s3_bucket.backups.id
}

# =============================================================================
# CloudWatch Outputs
# =============================================================================

output "log_group_name" {
  description = "CloudWatch log group name"
  value       = aws_cloudwatch_log_group.ecs.name
}

# =============================================================================
# Connection Information
# =============================================================================

output "connection_string" {
  description = "Database connection string (for .env file)"
  value       = "postgresql://${var.db_username}:${var.db_password}@${aws_db_instance.postgres.address}:${aws_db_instance.postgres.port}/${aws_db_instance.postgres.db_name}"
  sensitive   = true
}

output "environment_variables" {
  description = "Environment variables for application configuration"
  value = {
    ENVIRONMENT       = var.environment
    AWS_REGION        = var.aws_region
    DB_HOST           = aws_db_instance.postgres.address
    DB_PORT           = aws_db_instance.postgres.port
    DB_NAME           = aws_db_instance.postgres.db_name
    DB_USER           = var.db_username
    MODELS_BUCKET     = aws_s3_bucket.models.id
    BACKUPS_BUCKET    = aws_s3_bucket.backups.id
    CLOUDWATCH_REGION = var.aws_region
  }
  sensitive = true
}

# =============================================================================
# Deployment Instructions
# =============================================================================

output "deployment_instructions" {
  description = "Next steps for deployment"
  value = <<-EOT
    ✅ Infrastructure created successfully!
    
    Next steps:
    1. Push Docker image to ECR:
       aws ecr get-login-password --region ${var.aws_region} | docker login --username AWS --password-stdin ${var.ecr_repository_url}
       docker tag alpharl-quant:latest ${var.ecr_repository_url}:${var.image_tag}
       docker push ${var.ecr_repository_url}:${var.image_tag}
    
    2. Update ECS service:
       aws ecs update-service --cluster ${aws_ecs_cluster.main.name} --service ${aws_ecs_service.pipeline.name} --force-new-deployment
    
    3. View logs:
       aws logs tail ${aws_cloudwatch_log_group.ecs.name} --follow
    
    4. Database connection:
       Host: ${aws_db_instance.postgres.address}
       Port: ${aws_db_instance.postgres.port}
       Database: ${aws_db_instance.postgres.db_name}
       User: ${var.db_username}
  EOT
}
