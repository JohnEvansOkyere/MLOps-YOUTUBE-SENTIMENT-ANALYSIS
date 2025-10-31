# monitoring/model_monitor.py
"""
Model Performance Monitoring using Evidently AI
"""

import pandas as pd
import numpy as np
from pathlib import Path
import yaml
import logging
from datetime import datetime
from typing import Dict, Tuple

from evidently.report import Report
from evidently.metric_preset import ClassificationPreset
from evidently.metrics import *

logger = logging.getLogger(__name__)


class ModelPerformanceMonitor:
    """Monitor model performance and detect degradation."""
    
    def __init__(self, config_path: str = "params.yaml"):
        """Initialize performance monitor."""
        self.config = self._load_config(config_path)
        self.accuracy_threshold = self.config['monitoring']['performance']['accuracy_threshold']
        self.f1_threshold = self.config['monitoring']['performance']['f1_threshold']
        self.degradation_threshold = self.config['monitoring']['performance']['degradation_threshold']
    
    def _load_config(self, config_path: str) -> Dict:
        """Load monitoring configuration."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def monitor_performance(
        self,
        reference_data: pd.DataFrame,
        current_data: pd.DataFrame,
        target_col: str = 'category',
        prediction_col: str = 'prediction',
        save_report: bool = True
    ) -> Tuple[bool, Dict]:
        """
        Monitor model performance and detect degradation.
        
        Args:
            reference_data: Reference dataset with true labels and predictions
            current_data: Current dataset with true labels and predictions
            target_col: Name of the target column
            prediction_col: Name of the prediction column
            save_report: Whether to save the report
            
        Returns:
            - degradation_detected: bool
            - performance_summary: dict
        """
        logger.info("Starting model performance monitoring...")
        
        # Create performance report
        report = Report(metrics=[
            ClassificationPreset(),
        ])
        
        # Run report
        report.run(
            reference_data=reference_data,
            current_data=current_data,
            column_mapping={
                'target': target_col,
                'prediction': prediction_col
            }
        )
        
        # Extract metrics
        results = report.as_dict()
        
        # Get accuracy and F1 from current data
        current_metrics = self._extract_metrics(results, 'current')
        reference_metrics = self._extract_metrics(results, 'reference')
        
        # Check for degradation
        accuracy_drop = reference_metrics['accuracy'] - current_metrics['accuracy']
        f1_drop = reference_metrics['f1_weighted'] - current_metrics['f1_weighted']
        
        degradation_detected = (
            current_metrics['accuracy'] < self.accuracy_threshold or
            current_metrics['f1_weighted'] < self.f1_threshold or
            accuracy_drop > self.degradation_threshold or
            f1_drop > self.degradation_threshold
        )
        
        performance_summary = {
            'degradation_detected': degradation_detected,
            'current_metrics': current_metrics,
            'reference_metrics': reference_metrics,
            'accuracy_drop': accuracy_drop,
            'f1_drop': f1_drop,
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"Performance Monitoring Results:")
        logger.info(f"  Current Accuracy: {current_metrics['accuracy']:.4f}")
        logger.info(f"  Reference Accuracy: {reference_metrics['accuracy']:.4f}")
        logger.info(f"  Accuracy Drop: {accuracy_drop:.4f}")
        logger.info(f"  Degradation Detected: {degradation_detected}")
        
        # Save report
        if save_report:
            reports_dir = Path("reports") / "performance"
            reports_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            html_path = reports_dir / f"performance_{timestamp}.html"
            report.save_html(str(html_path))
            logger.info(f"  Performance report saved: {html_path}")
        
        return degradation_detected, performance_summary
    
    def _extract_metrics(self, results: Dict, data_type: str) -> Dict:
        """Extract metrics from Evidently results."""
        metrics = {}
        
        for metric in results['metrics']:
            if metric['metric'] == 'ClassificationQualityMetric':
                result = metric['result'][data_type]
                metrics['accuracy'] = result['accuracy']
                metrics['precision'] = result['precision']
                metrics['recall'] = result['recall']
                metrics['f1_weighted'] = result['f1']
                break
        
        return metrics