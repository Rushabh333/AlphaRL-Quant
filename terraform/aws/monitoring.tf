# CloudWatch Monitoring, Dashboards, and Alarms

# =============================================================================
# CloudWatch Dashboard
# =============================================================================

resource "aws_cloudwatch_dashboard" "alpharl" {
  dashboard_name = "AlphaRL-${var.environment}"
  
  dashboard_body = jsonencode({
    widgets = [
      # ECS CPU and Memory
      {
        type = "metric"
        properties = {
          metrics = [
            ["AWS/ECS", "CPUUtilization", "ServiceName", aws_ecs_service.pipeline.name, "ClusterName", aws_ecs_cluster.main.name],
            [".", "MemoryUtilization", ".", ".", ".", "."],
          ]
          period = 300
          stat   = "Average"
          region = var.aws_region
          title  = "ECS Service - CPU & Memory"
          yAxis = {
            left = {
              min = 0
              max = 100
            }
          }
        }
      },
      
      # ECS Task Count
      {
        type = "metric"
        properties = {
          metrics = [
            ["AWS/ECS", "DesiredTaskCount", "ServiceName", aws_ecs_service.pipeline.name, "ClusterName", aws_ecs_cluster.main.name],
            [".", "RunningTaskCount", ".", ".", ".", "."],
          ]
          period = 300
          stat   = "Average"
          region = var.aws_region
          title  = "ECS Service - Task Count"
        }
      },
      
      # RDS CPU and Connections
      {
        type = "metric"
        properties = {
          metrics = [
            ["AWS/RDS", "CPUUtilization", "DBInstanceIdentifier", aws_db_instance.postgres.id],
            [".", "DatabaseConnections", ".", "."],
          ]
          period = 300
          stat   = "Average"
          region = var.aws_region
          title  = "RDS - CPU & Connections"
        }
      },
      
      # RDS Storage and IOPS
      {
        type = "metric"
        properties = {
          metrics = [
            ["AWS/RDS", "FreeStorageSpace", "DBInstanceIdentifier", aws_db_instance.postgres.id],
            [".", "ReadIOPS", ".", "."],
            [".", "WriteIOPS", ".", "."],
          ]
          period = 300
          stat   = "Average"
          region = var.aws_region
          title  = "RDS - Storage & IOPS"
        }
      },
      
      # Application Logs
      {
        type = "log"
        properties = {
          query   = "SOURCE '${aws_cloudwatch_log_group.ecs.name}' | fields @timestamp, @message | filter @message like /ERROR/ | sort @timestamp desc | limit 20"
          region  = var.aws_region
          title   = "Recent Errors"
        }
      },
      
      # Custom Metrics (Trading Performance)
      {
        type = "metric"
        properties = {
          metrics = [
            ["AlphaRL", "SharpeRatio", "Environment", var.environment],
            [".", "PortfolioValue", ".", "."],
          ]
          period = 3600
          stat   = "Average"
          region = var.aws_region
          title  = "Trading Performance"
        }
      }
    ]
  })
}

# =============================================================================
# CloudWatch Alarms - ECS Service
# =============================================================================

resource "aws_cloudwatch_metric_alarm" "ecs_cpu_high" {
  alarm_name          = "alpharl-ecs-cpu-high-${var.environment}"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 2
  metric_name         = "CPUUtilization"
  namespace           = "AWS/ECS"
  period              = 300
  statistic           = "Average"
  threshold           = 85
  alarm_description   = "ECS CPU utilization is too high"
  alarm_actions       = [aws_sns_topic.alerts.arn]
  
  dimensions = {
    ClusterName = aws_ecs_cluster.main.name
    ServiceName = aws_ecs_service.pipeline.name
  }
  
  tags = {
    Name = "alpharl-ecs-cpu-high"
  }
}

resource "aws_cloudwatch_metric_alarm" "ecs_memory_high" {
  alarm_name          = "alpharl-ecs-memory-high-${var.environment}"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 2
  metric_name         = "MemoryUtilization"
  namespace           = "AWS/ECS"
  period              = 300
  statistic           = "Average"
  threshold           = 90
  alarm_description   = "ECS memory utilization is too high"
  alarm_actions       = [aws_sns_topic.alerts.arn]
  
  dimensions = {
    ClusterName = aws_ecs_cluster.main.name
    ServiceName = aws_ecs_service.pipeline.name
  }
  
  tags = {
    Name = "alpharl-ecs-memory-high"
  }
}

resource "aws_cloudwatch_metric_alarm" "ecs_tasks_stopped" {
  alarm_name          = "alpharl-ecs-tasks-stopped-${var.environment}"
  comparison_operator = "LessThanThreshold"
  evaluation_periods  = 1
  metric_name         = "RunningTaskCount"
  namespace           = "AWS/ECS"
  period              = 60
  statistic           = "Average"
  threshold           = 1
  alarm_description   = "No ECS tasks are running"
  alarm_actions       = [aws_sns_topic.alerts.arn]
  treat_missing_data  = "breaching"
  
  dimensions = {
    ClusterName = aws_ecs_cluster.main.name
    ServiceName = aws_ecs_service.pipeline.name
  }
  
  tags = {
    Name = "alpharl-ecs-tasks-stopped"
    Severity = "critical"
  }
}

# =============================================================================
# CloudWatch Alarms - RDS Database
# =============================================================================

resource "aws_cloudwatch_metric_alarm" "rds_cpu_high" {
  alarm_name          = "alpharl-rds-cpu-high-${var.environment}"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 2
  metric_name         = "CPUUtilization"
  namespace           = "AWS/RDS"
  period              = 300
  statistic           = "Average"
  threshold           = 80
  alarm_description   = "RDS CPU utilization is too high"
  alarm_actions       = [aws_sns_topic.alerts.arn]
  
  dimensions = {
    DBInstanceIdentifier = aws_db_instance.postgres.id
  }
  
  tags = {
    Name = "alpharl-rds-cpu-high"
  }
}

resource "aws_cloudwatch_metric_alarm" "rds_storage_low" {
  alarm_name          = "alpharl-rds-storage-low-${var.environment}"
  comparison_operator = "LessThanThreshold"
  evaluation_periods  = 1
  metric_name         = "FreeStorageSpace"
  namespace           = "AWS/RDS"
  period              = 300
  statistic           = "Average"
  threshold           = 5000000000  # 5 GB
  alarm_description   = "RDS free storage space is low"
  alarm_actions       = [aws_sns_topic.alerts.arn]
  
  dimensions = {
    DBInstanceIdentifier = aws_db_instance.postgres.id
  }
  
  tags = {
    Name     = "alpharl-rds-storage-low"
    Severity = "warning"
  }
}

resource "aws_cloudwatch_metric_alarm" "rds_connections_high" {
  alarm_name          = "alpharl-rds-connections-high-${var.environment}"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 2
  metric_name         = "DatabaseConnections"
  namespace           = "AWS/RDS"
  period              = 300
  statistic           = "Average"
  threshold           = 80
  alarm_description   = "RDS connection count is too high"
  alarm_actions       = [aws_sns_topic.alerts.arn]
  
  dimensions = {
    DBInstanceIdentifier = aws_db_instance.postgres.id
  }
  
  tags = {
    Name = "alpharl-rds-connections-high"
  }
}

# =============================================================================
# CloudWatch Alarms - Trading Performance
# =============================================================================

resource "aws_cloudwatch_metric_alarm" "high_drawdown" {
  alarm_name          = "alpharl-high-drawdown-${var.environment}"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 1
  metric_name         = "PortfolioDrawdown"
  namespace           = "AlphaRL"
  period              = 300
  statistic           = "Maximum"
  threshold           = 10  # 10% drawdown
  alarm_description   = "Portfolio drawdown exceeded threshold"
  alarm_actions       = [aws_sns_topic.critical_alerts.arn]
  
  dimensions = {
    Environment = var.environment
  }
  
  tags = {
    Name     = "alpharl-high-drawdown"
    Severity = "critical"
  }
}

resource "aws_cloudwatch_metric_alarm" "model_degradation" {
  alarm_name          = "alpharl-model-degradation-${var.environment}"
  comparison_operator = "LessThanThreshold"
  evaluation_periods  = 3
  metric_name         = "SharpeRatio"
  namespace           = "AlphaRL"
  period              = 3600
  statistic           = "Average"
  threshold           = 0.5
  alarm_description   = "Model performance (Sharpe ratio) has degraded"
  alarm_actions       = [aws_sns_topic.alerts.arn]
  
  dimensions = {
    Environment = var.environment
  }
  
  tags = {
    Name     = "alpharl-model-degradation"
    Severity = "warning"
  }
}

resource "aws_cloudwatch_metric_alarm" "api_error_rate" {
  alarm_name          = "alpharl-api-error-rate-${var.environment}"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 2
  metric_name         = "APIErrorRate"
  namespace           = "AlphaRL"
  period              = 300
  statistic           = "Average"
  threshold           = 10  # 10% error rate
  alarm_description   = "API error rate is too high"
  alarm_actions       = [aws_sns_topic.alerts.arn]
  
  dimensions = {
    Environment = var.environment
  }
  
  tags = {
    Name = "alpharl-api-error-rate"
  }
}

# =============================================================================
# SNS Topics for Alerts
# =============================================================================

resource "aws_sns_topic" "alerts" {
  name = "alpharl-alerts-${var.environment}"
  
  tags = {
    Name = "alpharl-alerts"
  }
}

resource "aws_sns_topic" "critical_alerts" {
  name = "alpharl-critical-alerts-${var.environment}"
  
  tags = {
    Name     = "alpharl-critical-alerts"
    Severity = "critical"
  }
}

resource "aws_sns_topic_subscription" "alerts_email" {
  count = var.alarm_email != "" ? 1 : 0
  
  topic_arn = aws_sns_topic.alerts.arn
  protocol  = "email"
  endpoint  = var.alarm_email
}

resource "aws_sns_topic_subscription" "critical_alerts_email" {
  count = var.alarm_email != "" ? 1 : 0
  
  topic_arn = aws_sns_topic.critical_alerts.arn
  protocol  = "email"
  endpoint  = var.alarm_email
}

# =============================================================================
# CloudWatch Log Metric Filters
# =============================================================================

# Track ERROR logs
resource "aws_cloudwatch_log_metric_filter" "errors" {
  name           = "alpharl-errors-${var.environment}"
  log_group_name = aws_cloudwatch_log_group.ecs.name
  pattern        = "[time, request_id, level = ERROR*, ...]"
  
  metric_transformation {
    name      = "ErrorCount"
    namespace = "AlphaRL"
    value     = "1"
    dimensions = {
      Environment = var.environment
    }
  }
}

# Create alarm on error count
resource "aws_cloudwatch_metric_alarm" "error_count" {
  alarm_name          = "alpharl-error-count-${var.environment}"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 1
  metric_name         = "ErrorCount"
  namespace           = "AlphaRL"
  period              = 300
  statistic           = "Sum"
  threshold           = 10
  alarm_description   = "Too many errors in logs"
  alarm_actions       = [aws_sns_topic.alerts.arn]
  treat_missing_data  = "notBreaching"
  
  dimensions = {
    Environment = var.environment
  }
  
  tags = {
    Name = "alpharl-error-count"
  }
}
