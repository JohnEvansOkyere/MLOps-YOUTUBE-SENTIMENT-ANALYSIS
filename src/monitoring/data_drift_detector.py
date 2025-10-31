# monitoring/data_drift_detector.py
"""
Data Drift Detection using Evidently AI
Compares current data against reference (training) data
"""

import pandas as pd
import numpy as np
from pathlib import Path
import yaml
import logging
from datetime import datetime
from typing import Dict, Tuple, Optional

from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, DataQualityPreset
from evidently.metrics import *
from evidently.test_suite import TestSuite
from evidently.tests import *

logger = logging.getLogger(__name__)


class DataDriftDetector:
    """Detect data drift between reference and current datasets."""
    
    def __init__(self, config_path: str = "params.yaml"):
        """Initialize drift detector with configuration."""
        self.config = self._load_config(config_path)
        # Now access monitoring section
        self.drift_threshold = self.config['monitoring']['data_drift']['threshold']
        
    def _load_config(self, config_path: str) -> Dict:
        """Load monitoring configuration."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def prepare_data(
        self, 
        df: pd.DataFrame, 
        text_column: str = 'clean_comment'
    ) -> pd.DataFrame:
        """Prepare data for drift detection by adding statistical features."""
        df = df.copy()
        
        # Text length features
        df['text_length'] = df[text_column].str.len()
        df['word_count'] = df[text_column].str.split().str.len()
        df['avg_word_length'] = df['text_length'] / (df['word_count'] + 1)
        
        # Character-based features
        df['uppercase_ratio'] = df[text_column].str.count('[A-Z]') / (df['text_length'] + 1)
        df['digit_ratio'] = df[text_column].str.count('[0-9]') / (df['text_length'] + 1)
        df['special_char_ratio'] = df[text_column].str.count('[^a-zA-Z0-9\s]') / (df['text_length'] + 1)
        
        # Sentiment indicators (simple heuristics)
        df['exclamation_count'] = df[text_column].str.count('!')
        df['question_count'] = df[text_column].str.count('\?')
        df['emoji_count'] = df[text_column].str.count('[\U0001F600-\U0001F64F]')
        
        return df
    
    def detect_drift(
        self,
        reference_data: pd.DataFrame,
        current_data: pd.DataFrame,
        save_report: bool = True
    ) -> Tuple[bool, Dict, Report]:
        """
        Detect data drift between reference and current data.
        
        Returns:
            - drift_detected: bool
            - drift_summary: dict with metrics
            - report: Evidently Report object
        """
        logger.info("Starting data drift detection...")
        
        # Prepare both datasets
        reference_prep = self.prepare_data(reference_data)
        current_prep = self.prepare_data(current_data)
        
        # Define columns to monitor
        numerical_features = [
            'text_length', 'word_count', 'avg_word_length',
            'uppercase_ratio', 'digit_ratio', 'special_char_ratio',
            'exclamation_count', 'question_count', 'emoji_count'
        ]
        
        # Create drift report
        report = Report(metrics=[
            DataDriftPreset(
                stattest='wasserstein',
                stattest_threshold=self.drift_threshold
            ),
            DataQualityPreset(),
        ])
        
        # Run the report
        report.run(
            reference_data=reference_prep[numerical_features],
            current_data=current_prep[numerical_features]
        )
        
        # Extract drift results
        drift_results = report.as_dict()
        
        # Check if drift detected
        drift_detected = drift_results['metrics'][0]['result']['dataset_drift']
        n_drifted_features = drift_results['metrics'][0]['result']['number_of_drifted_columns']
        
        drift_summary = {
            'drift_detected': drift_detected,
            'n_drifted_features': n_drifted_features,
            'drift_score': drift_results['metrics'][0]['result']['share_of_drifted_columns'],
            'timestamp': datetime.now().isoformat(),
            'reference_size': len(reference_data),
            'current_size': len(current_data)
        }
        
        logger.info(f"Drift Detection Results:")
        logger.info(f"  Drift Detected: {drift_detected}")
        logger.info(f"  Drifted Features: {n_drifted_features}")
        logger.info(f"  Drift Score: {drift_summary['drift_score']:.3f}")
        
        # Save report
        if save_report:
            self._save_report(report, drift_summary, "drift")
        
        return drift_detected, drift_summary, report
    
    def _save_report(self, report: Report, summary: Dict, report_type: str):
        """Save Evidently report to file."""
        reports_dir = Path("reports") / report_type
        reports_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save HTML report
        html_path = reports_dir / f"report_{timestamp}.html"
        report.save_html(str(html_path))
        logger.info(f"  HTML report saved: {html_path}")
        
        # Save JSON report
        json_path = reports_dir / f"report_{timestamp}.json"
        report.save_json(str(json_path))
        logger.info(f"  JSON report saved: {json_path}")
        
        return html_path, json_path


class DataQualityChecker:
    """Check data quality using Evidently."""
    
    def __init__(self, config_path: str = "monitoring/config/monitoring_config.yaml"):
        """Initialize quality checker."""
        self.config = self._load_config(config_path)
        
    def _load_config(self, config_path: str) -> Dict:
        """Load monitoring configuration."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def check_quality(
        self,
        data: pd.DataFrame,
        text_column: str = 'clean_comment',
        save_report: bool = True
    ) -> Tuple[bool, Dict]:
        """
        Check data quality.
        
        Returns:
            - quality_passed: bool
            - quality_summary: dict
        """
        logger.info("Starting data quality check...")
        
        # Create test suite
        test_suite = TestSuite(tests=[
            TestNumberOfRows(gt=10),  # At least 10 rows
            TestNumberOfMissingValues(),
            TestNumberOfDuplicatedRows(),
            TestColumnsType(),
        ])
        
        # Run tests
        test_suite.run(reference_data=None, current_data=data)
        
        # Get results
        results = test_suite.as_dict()
        
        # Count passed/failed tests
        n_tests = len(results['tests'])
        n_passed = sum(1 for test in results['tests'] if test['status'] == 'SUCCESS')
        n_failed = n_tests - n_passed
        
        quality_passed = n_failed == 0
        
        quality_summary = {
            'quality_passed': quality_passed,
            'n_tests': n_tests,
            'n_passed': n_passed,
            'n_failed': n_failed,
            'timestamp': datetime.now().isoformat(),
            'data_size': len(data)
        }
        
        logger.info(f"Data Quality Results:")
        logger.info(f"  Tests Passed: {n_passed}/{n_tests}")
        logger.info(f"  Quality OK: {quality_passed}")
        
        # Save report
        if save_report:
            reports_dir = Path("reports") / "data_quality"
            reports_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            html_path = reports_dir / f"quality_{timestamp}.html"
            test_suite.save_html(str(html_path))
            logger.info(f"  Quality report saved: {html_path}")
        
        return quality_passed, quality_summary