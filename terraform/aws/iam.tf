# IAM Roles and Policies for AlphaRL-Quant

# =============================================================================
# ECS Execution Role (for pulling images and writing logs)
# =============================================================================

resource "aws_iam_role" "ecs_execution" {
  name = "alpharl-ecs-execution-${var.environment}"
  
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "ecs-tasks.amazonaws.com"
        }
      }
    ]
  })
  
  tags = {
    Name = "alpharl-ecs-execution"
  }
}

resource "aws_iam_role_policy_attachment" "ecs_execution_policy" {
  role       = aws_iam_role.ecs_execution.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

# Additional policy for Secrets Manager access
resource "aws_iam_role_policy" "ecs_secrets" {
  name = "alpharl-ecs-secrets"
  role = aws_iam_role.ecs_execution.id
  
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "secretsmanager:GetSecretValue",
          "kms:Decrypt"
        ]
        Resource = [
          aws_secretsmanager_secret.db_password.arn,
          aws_secretsmanager_secret.api_keys.arn
        ]
      }
    ]
  })
}

# =============================================================================
# ECS Task Role (for application permissions)
# =============================================================================

resource "aws_iam_role" "ecs_task" {
  name = "alpharl-ecs-task-${var.environment}"
  
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "ecs-tasks.amazonaws.com"
        }
      }
    ]
  })
  
  tags = {
    Name = "alpharl-ecs-task"
  }
}

# S3 access for models and backups
resource "aws_iam_role_policy" "ecs_s3" {
  name = "alpharl-ecs-s3"
  role = aws_iam_role.ecs_task.id
  
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:DeleteObject",
          "s3:ListBucket"
        ]
        Resource = [
          aws_s3_bucket.models.arn,
          "${aws_s3_bucket.models.arn}/*",
          aws_s3_bucket.backups.arn,
          "${aws_s3_bucket.backups.arn}/*"
        ]
      }
    ]
  })
}

# CloudWatch metrics and logs
resource "aws_iam_role_policy" "ecs_cloudwatch" {
  name = "alpharl-ecs-cloudwatch"
  role = aws_iam_role.ecs_task.id
  
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "cloudwatch:PutMetricData",
          "logs:CreateLogStream",
          "logs:PutLogEvents"
        ]
        Resource = "*"
      }
    ]
  })
}

# =============================================================================
# RDS Monitoring Role
# =============================================================================

resource "aws_iam_role" "rds_monitoring" {
  name = "alpharl-rds-monitoring-${var.environment}"
  
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "monitoring.rds.amazonaws.com"
        }
      }
    ]
  })
  
  tags = {
    Name = "alpharl-rds-monitoring"
  }
}

resource "aws_iam_role_policy_attachment" "rds_monitoring_policy" {
  role       = aws_iam_role.rds_monitoring.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonRDSEnhancedMonitoringRole"
}

# =============================================================================
# S3 Replication Role (for disaster recovery)
# =============================================================================

resource "aws_iam_role" "s3_replication" {
  count = var.enable_disaster_recovery ? 1 : 0
  name  = "alpharl-s3-replication-${var.environment}"
  
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "s3.amazonaws.com"
        }
      }
    ]
  })
  
  tags = {
    Name = "alpharl-s3-replication"
  }
}

resource "aws_iam_role_policy" "s3_replication" {
  count = var.enable_disaster_recovery ? 1 : 0
  name  = "alpharl-s3-replication-policy"
  role  = aws_iam_role.s3_replication[0].id
  
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:GetReplicationConfiguration",
          "s3:ListBucket"
        ]
        Resource = aws_s3_bucket.backups.arn
      },
      {
        Effect = "Allow"
        Action = [
          "s3:GetObjectVersionForReplication",
          "s3:GetObjectVersionAcl"
        ]
        Resource = "${aws_s3_bucket.backups.arn}/*"
      },
      {
        Effect = "Allow"
        Action = [
          "s3:ReplicateObject",
          "s3:ReplicateDelete"
        ]
        Resource = "${aws_s3_bucket.backups_replica[0].arn}/*"
      }
    ]
  })
}

# =============================================================================
# Lambda Execution Role (for automated tasks)
# =============================================================================

resource "aws_iam_role" "lambda_execution" {
  name = "alpharl-lambda-execution-${var.environment}"
  
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "lambda.amazonaws.com"
        }
      }
    ]
  })
  
  tags = {
    Name = "alpharl-lambda-execution"
  }
}

resource "aws_iam_role_policy_attachment" "lambda_basic" {
  role       = aws_iam_role.lambda_execution.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

# =============================================================================
# Secrets Manager
# =============================================================================

resource "aws_secretsmanager_secret" "db_password" {
  name        = "alpharl/${var.environment}/db-password"
  description = "Database password for AlphaRL-Quant"
  
  tags = {
    Name = "alpharl-db-password"
  }
}

resource "aws_secretsmanager_secret_version" "db_password" {
  secret_id     = aws_secretsmanager_secret.db_password.id
  secret_string = var.db_password
}

resource "aws_secretsmanager_secret" "api_keys" {
  name        = "alpharl/${var.environment}/api-keys"
  description = "API keys for AlphaRL-Quant"
  
  tags = {
    Name = "alpharl-api-keys"
  }
}

# =============================================================================
# CloudWatch Log Group
# =============================================================================

resource "aws_cloudwatch_log_group" "ecs" {
  name              = "/ecs/alpharl-${var.environment}"
  retention_in_days = var.environment == "production" ? 90 : 7
  
  tags = {
    Name = "alpharl-ecs-logs"
  }
}

# =============================================================================
# Disaster Recovery Resources (in alternate region)
# =============================================================================

# DR S3 bucket in alternate region
resource "aws_s3_bucket" "backups_replica" {
  count    = var.enable_disaster_recovery ? 1 : 0
  provider = aws.dr  # Requires separate provider
  
  bucket = "alpharl-backups-replica-${var.environment}-${data.aws_caller_identity.current.account_id}"
  
  tags = {
    Name    = "alpharl-backups-replica"
    Purpose = "disaster-recovery"
  }
}
