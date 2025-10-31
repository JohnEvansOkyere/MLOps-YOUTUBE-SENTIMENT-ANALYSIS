# monitoring/report_generator.py
"""
Generate comprehensive monitoring reports
"""

import mlflow
from pathlib import Path
import logging
from typing import Dict
import json

logger = logging.getLogger(__name__)


class MonitoringReportGenerator:
    """Generate and log monitoring reports to MLflow."""
    
    def __init__(self):
        self.reports_dir = Path("reports")
        
    def log_to_mlflow(
        self,
        drift_summary: Dict,
        quality_summary: Dict,
        performance_summary: Dict = None,
        report_paths: Dict = None
    ):
        """Log monitoring metrics and reports to MLflow."""
        try:
            # Log drift metrics
            mlflow.log_metric("drift_detected", int(drift_summary['drift_detected']))
            mlflow.log_metric("drift_score", drift_summary['drift_score'])
            mlflow.log_metric("n_drifted_features", drift_summary['n_drifted_features'])
            
            # Log quality metrics
            mlflow.log_metric("quality_passed", int(quality_summary['quality_passed']))
            mlflow.log_metric("quality_tests_passed", quality_summary['n_passed'])
            mlflow.log_metric("quality_tests_failed", quality_summary['n_failed'])
            
            # Log performance metrics if available
            if performance_summary:
                mlflow.log_metric("performance_degradation", int(performance_summary['degradation_detected']))
                mlflow.log_metric("accuracy_drop", performance_summary['accuracy_drop'])
                mlflow.log_metric("f1_drop", performance_summary['f1_drop'])
            
            # Log report files as artifacts
            if report_paths:
                for report_type, paths in report_paths.items():
                    for path in paths:
                        if Path(path).exists():
                            mlflow.log_artifact(str(path), artifact_path=f"monitoring/{report_type}")
            
            # Save summary as JSON
            summary = {
                'drift': drift_summary,
                'quality': quality_summary,
                'performance': performance_summary
            }
            
            summary_path = self.reports_dir / "monitoring_summary.json"
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2)
            
            mlflow.log_artifact(str(summary_path), artifact_path="monitoring")
            
            logger.info("✓ Monitoring metrics logged to MLflow")
            
        except Exception as e:
            logger.error(f"Failed to log monitoring metrics to MLflow: {e}")