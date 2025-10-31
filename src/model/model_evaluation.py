# src/model/model_evaluation.py
"""
Model Evaluation Script with Evidently AI Monitoring Integration
Evaluates trained model, detects drift, monitors performance, and logs to MLflow
"""

import numpy as np
import pandas as pd
import pickle
import logging
import yaml
import mlflow
import mlflow.sklearn
import os
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple
from dotenv import load_dotenv

from sklearn.metrics import (
    classification_report, 
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from mlflow.models import infer_signature

# Add project root to path for imports
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

# Import monitoring modules
from src.monitoring.data_drift_detector import DataDriftDetector, DataQualityChecker
from src.monitoring.report_generator import MonitoringReportGenerator

# Load environment variables
load_dotenv()

# Configuration
MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI')
MLFLOW_EXPERIMENT_NAME = os.getenv('MLFLOW_EXPERIMENT_NAME', 'yt_sentiment_analysis')
ENVIRONMENT = os.getenv('ENVIRONMENT', 'development')
GIT_COMMIT_HASH = os.getenv('GIT_COMMIT_HASH', 'unknown')

# ==================== LOGGING SETUP ====================
def setup_logging() -> logging.Logger:
    """Configure logging with both console and file handlers."""
    logger = logging.getLogger('model_evaluation')
    logger.setLevel(logging.DEBUG)
    
    # Remove existing handlers
    logger.handlers = []
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Create logs directory
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)
    
    # File handler for errors
    file_handler = logging.FileHandler(log_dir / 'model_evaluation_errors.log')
    file_handler.setLevel(logging.ERROR)
    
    # Detailed file handler
    detailed_handler = logging.FileHandler(log_dir / 'model_evaluation.log')
    detailed_handler.setLevel(logging.DEBUG)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    detailed_handler.setFormatter(formatter)
    
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    logger.addHandler(detailed_handler)
    
    return logger

logger = setup_logging()

# ==================== UTILITY FUNCTIONS ====================
def get_root_directory() -> Path:
    """Get the root directory of the project."""
    return project_root

def load_params(params_path: str) -> Dict:
    """Load parameters from a YAML file."""
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.info(f'✓ Parameters loaded from {params_path}')
        return params
    except Exception as e:
        logger.error(f'Error loading parameters: {e}')
        raise

def load_data(file_path: str) -> pd.DataFrame:
    """Load data from a CSV file with validation."""
    try:
        if not Path(file_path).exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        df = pd.read_csv(file_path)
        df.fillna('', inplace=True)
        
        logger.info(f'✓ Data loaded from {file_path}')
        logger.info(f'  Shape: {df.shape}')
        
        return df
    except Exception as e:
        logger.error(f'Error loading data: {e}')
        raise

def load_model(model_path: str):
    """Load the trained model."""
    try:
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        logger.info(f'✓ Model loaded from {model_path}')
        return model
    except Exception as e:
        logger.error(f'Error loading model: {e}')
        raise

def load_vectorizer(vectorizer_path: str):
    """Load the TF-IDF vectorizer."""
    try:
        with open(vectorizer_path, 'rb') as file:
            vectorizer = pickle.load(file)
        logger.info(f'✓ Vectorizer loaded from {vectorizer_path}')
        return vectorizer
    except Exception as e:
        logger.error(f'Error loading vectorizer: {e}')
        raise

# ==================== EVALUATION FUNCTIONS ====================
def evaluate_model(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    dataset_name: str = "test"
) -> Tuple[Dict, np.ndarray, np.ndarray]:
    """
    Evaluate model and return comprehensive metrics.
    
    Returns:
        metrics: Dict of evaluation metrics
        y_pred: Predictions
        y_pred_proba: Prediction probabilities
    """
    try:
        logger.info(f"\n{'='*60}")
        logger.info(f"MODEL EVALUATION - {dataset_name.upper()}")
        logger.info(f"{'='*60}")
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision_weighted': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'recall_weighted': recall_score(y_test, y_pred, average='weighted', zero_division=0),
            'f1_score_weighted': f1_score(y_test, y_pred, average='weighted', zero_division=0),
            'precision_macro': precision_score(y_test, y_pred, average='macro', zero_division=0),
            'recall_macro': recall_score(y_test, y_pred, average='macro', zero_division=0),
            'f1_score_macro': f1_score(y_test, y_pred, average='macro', zero_division=0),
        }
        
        # Try to calculate ROC AUC for multiclass
        try:
            metrics['roc_auc_ovr'] = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='weighted')
            metrics['roc_auc_ovo'] = roc_auc_score(y_test, y_pred_proba, multi_class='ovo', average='weighted')
        except Exception as e:
            logger.warning(f"Could not calculate ROC AUC: {e}")
        
        # Log overall metrics
        logger.info("\nOverall Metrics:")
        logger.info(f"  Accuracy:           {metrics['accuracy']:.4f}")
        logger.info(f"  Precision (weighted): {metrics['precision_weighted']:.4f}")
        logger.info(f"  Recall (weighted):    {metrics['recall_weighted']:.4f}")
        logger.info(f"  F1 Score (weighted):  {metrics['f1_score_weighted']:.4f}")
        
        # Classification report
        logger.info("\nDetailed Classification Report:")
        report = classification_report(y_test, y_pred, zero_division=0)
        logger.info(f"\n{report}")
        
        # Get per-class metrics
        report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        
        # Add per-class metrics to main metrics dict
        for label, label_metrics in report_dict.items():
            if isinstance(label_metrics, dict) and label not in ['accuracy', 'macro avg', 'weighted avg']:
                metrics[f'class_{label}_precision'] = label_metrics['precision']
                metrics[f'class_{label}_recall'] = label_metrics['recall']
                metrics[f'class_{label}_f1'] = label_metrics['f1-score']
                metrics[f'class_{label}_support'] = label_metrics['support']
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        logger.info("\nConfusion Matrix:")
        logger.info(f"\n{cm}")
        
        return metrics, y_pred, y_pred_proba, cm, report_dict
        
    except Exception as e:
        logger.error(f'Error during model evaluation: {e}')
        raise

def save_confusion_matrix(
    cm: np.ndarray,
    dataset_name: str,
    class_names: list = None
) -> str:
    """Save confusion matrix plot and return file path."""
    try:
        plt.figure(figsize=(10, 8))
        
        if class_names:
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=class_names, yticklabels=class_names)
        else:
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        
        plt.title(f'Confusion Matrix - {dataset_name}', fontsize=14, pad=20)
        plt.xlabel('Predicted', fontsize=12)
        plt.ylabel('Actual', fontsize=12)
        plt.tight_layout()
        
        # Save to file
        output_dir = Path('reports') / 'confusion_matrices'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        cm_file_path = output_dir / f'confusion_matrix_{dataset_name}_{timestamp}.png'
        
        plt.savefig(cm_file_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f'✓ Confusion matrix saved to {cm_file_path}')
        return str(cm_file_path)
        
    except Exception as e:
        logger.error(f'Error saving confusion matrix: {e}')
        raise

def save_metrics_report(
    metrics: Dict,
    report_dict: Dict,
    dataset_name: str
) -> str:
    """Save detailed metrics report as JSON."""
    try:
        output_dir = Path('reports') / 'metrics'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = output_dir / f'metrics_{dataset_name}_{timestamp}.json'
        
        report_data = {
            'timestamp': datetime.now().isoformat(),
            'dataset': dataset_name,
            'environment': ENVIRONMENT,
            'git_commit': GIT_COMMIT_HASH,
            'overall_metrics': metrics,
            'classification_report': report_dict
        }
        
        with open(report_path, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        logger.info(f'✓ Metrics report saved to {report_path}')
        return str(report_path)
        
    except Exception as e:
        logger.error(f'Error saving metrics report: {e}')
        raise


def save_latest_artifacts(cm_path: str, metrics_path: str):
    """Create 'latest' copies of artifacts for DVC tracking."""
    try:
        import shutil
        reports_dir = Path('reports')
        
        # Copy confusion matrix to 'latest'
        cm_latest = reports_dir / 'confusion_matrices' / 'confusion_matrix_test_latest.png'
        cm_latest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(cm_path, cm_latest)
        
        # Copy metrics to 'latest'
        metrics_latest = reports_dir / 'metrics' / 'metrics_test_latest.json'
        metrics_latest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(metrics_path, metrics_latest)
        
        logger.info(f"✓ Latest artifacts created for DVC tracking")
        
    except Exception as e:
        logger.warning(f"Could not create latest artifacts: {e}")

# ==================== MONITORING INTEGRATION ====================
def run_drift_detection(
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
    config_path: str = "params.yaml"  # Add this parameter
) -> Tuple[bool, Dict]:
    """Run Evidently drift detection."""
    try:
        logger.info(f"\n{'='*60}")
        logger.info("DRIFT DETECTION")
        logger.info(f"{'='*60}")
        
        detector = DataDriftDetector(config_path=config_path)  # Pass config path
        drift_detected, drift_summary, drift_report = detector.detect_drift(
            reference_data=reference_data,
            current_data=current_data,
            save_report=True
        )
        
        return drift_detected, drift_summary
        
    except Exception as e:
        logger.error(f'Error during drift detection: {e}')
        raise

def run_quality_check(
    data: pd.DataFrame,
    config_path: str = "params.yaml"  # Add this parameter
) -> Tuple[bool, Dict]:
    """Run Evidently data quality check."""
    try:
        logger.info(f"\n{'='*60}")
        logger.info("DATA QUALITY CHECK")
        logger.info(f"{'='*60}")
        
        checker = DataQualityChecker(config_path=config_path)  # Pass config path
        quality_passed, quality_summary = checker.check_quality(
            data=data,
            save_report=True
        )
        
        return quality_passed, quality_summary
        
    except Exception as e:
        logger.error(f'Error during quality check: {e}')
        raise

# ==================== MLFLOW INTEGRATION ====================
def save_experiment_info(
    run_id: str,
    model_path: str,
    metrics: Dict,
    params: Dict,
    drift_summary: Dict,
    quality_summary: Dict
) -> Dict:
    """Save comprehensive experiment information."""
    try:
        root_dir = get_root_directory()
        
        experiment_info = {
            'run_id': run_id,
            'model_path': model_path,
            'model_type': 'lightgbm',
            'framework': 'lightgbm',
            'metrics': metrics,
            'parameters': params,
            'monitoring': {
                'drift': drift_summary,
                'quality': quality_summary
            },
            'timestamp': datetime.now().isoformat(),
            'git_commit': GIT_COMMIT_HASH,
            'environment': ENVIRONMENT,
            'dataset_version': 'v1.0',
        }
        
        # Save to root directory
        info_path = root_dir / 'experiment_info.json'
        with open(info_path, 'w') as f:
            json.dump(experiment_info, f, indent=2)
        
        logger.info(f"✓ Experiment info saved to {info_path}")
        
        return experiment_info
        
    except Exception as e:
        logger.error(f'Error saving experiment info: {e}')
        raise

# ==================== MAIN EVALUATION FLOW ====================
def main():
    """Main evaluation pipeline with monitoring integration."""
    
    try:
        logger.info(f"\n{'='*60}")
        logger.info("YOUTUBE SENTIMENT ANALYSIS - MODEL EVALUATION")
        logger.info(f"{'='*60}")
        logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Environment: {ENVIRONMENT}")
        logger.info(f"Git commit: {GIT_COMMIT_HASH}")
        logger.info(f"{'='*60}\n")
        
        # Get root directory
        root_dir = get_root_directory()
        
        # Set MLflow tracking
        if MLFLOW_TRACKING_URI:
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            logger.info(f"✓ MLflow tracking URI: {MLFLOW_TRACKING_URI}")
        
        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
        
        # Start MLflow run
        with mlflow.start_run(run_name=f"evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}") as run:
            run_id = run.info.run_id
            logger.info(f"MLflow Run ID: {run_id}\n")
            
            # Load parameters
            params_path = root_dir / 'params.yaml'
            params = load_params(str(params_path))
            
            # Log parameters
            model_params = params.get('model_building', {})
            for key, value in model_params.items():
                mlflow.log_param(key, value)
            
            mlflow.log_param('environment', ENVIRONMENT)
            mlflow.log_param('git_commit', GIT_COMMIT_HASH)
            
            # Load model and vectorizer
            logger.info("\n[1/7] Loading model and vectorizer...")
            model = load_model(str(root_dir / 'lgbm_model.pkl'))
            vectorizer = load_vectorizer(str(root_dir / 'tfidf_vectorizer.pkl'))
            
            # Load test data
            logger.info("\n[2/7] Loading test data...")
            test_data = load_data(str(root_dir / 'data/interim/test_processed.csv'))
            
            # Prepare test data
            X_test_text = test_data['clean_comment'].values
            X_test_tfidf = vectorizer.transform(X_test_text)
            y_test = test_data['category'].values
            
            logger.info(f"Test data shape: {X_test_tfidf.shape}")
            logger.info(f"Test labels shape: {y_test.shape}")
            
            # Evaluate model
            logger.info("\n[3/7] Evaluating model...")
            metrics, y_pred, y_pred_proba, cm, report_dict = evaluate_model(
                model, X_test_tfidf, y_test, "test"
            )
            
            # Log metrics to MLflow
            for metric_name, metric_value in metrics.items():
                if isinstance(metric_value, (int, float)):
                    mlflow.log_metric(f"test_{metric_name}", metric_value)
            
            # Save confusion matrix
            cm_path = save_confusion_matrix(cm, "test")
            mlflow.log_artifact(cm_path, artifact_path="visualizations")
            
            # Save metrics report
            metrics_report_path = save_metrics_report(metrics, report_dict, "test")
            mlflow.log_artifact(metrics_report_path, artifact_path="reports")
            
            save_latest_artifacts(cm_path, metrics_report_path)
            # Create DataFrame with predictions for monitoring
            test_results = test_data.copy()
            test_results['prediction'] = y_pred
            
            # Get full path to params.yaml
            params_config_path = str(root_dir / 'params.yaml')

            # Run data quality check
            logger.info("\n[4/7] Running data quality check...")
            quality_passed, quality_summary = run_quality_check(test_data, config_path=params_config_path)
            
            # Load reference data for drift detection
            logger.info("\n[5/7] Loading reference data for drift detection...")
            reference_data_path = root_dir / 'reference_data/train_reference.csv'
            
            if reference_data_path.exists():
                reference_data = load_data(str(reference_data_path))
                
                # Run drift detection
                logger.info("\n[6/7] Running drift detection...")
                drift_detected, drift_summary = run_drift_detection(
                    reference_data=reference_data,
                    current_data=test_data,
                    config_path=params_config_path
                )
            else:
                logger.warning(f"Reference data not found at {reference_data_path}")
                logger.warning("Skipping drift detection. Run: python scripts/create_reference_data.py")
                drift_detected = False
                drift_summary = {'drift_detected': False, 'message': 'Reference data not available'}
            
            # Log monitoring metrics to MLflow
            logger.info("\n[7/7] Logging monitoring metrics to MLflow...")
            report_gen = MonitoringReportGenerator()
            report_gen.log_to_mlflow(
                drift_summary=drift_summary,
                quality_summary=quality_summary
            )
            
            # Log model with signature
            input_example = pd.DataFrame(
                X_test_tfidf[:5].toarray(),
                columns=[f'feature_{i}' for i in range(X_test_tfidf.shape[1])]
            )
            signature = infer_signature(input_example, model.predict(X_test_tfidf[:5]))
            
            mlflow.sklearn.log_model(
                model,
                "model",
                signature=signature,
                input_example=input_example
            )
            
            # Log vectorizer
            mlflow.log_artifact(str(root_dir / 'tfidf_vectorizer.pkl'), artifact_path="vectorizer")
            
            # Set tags
            mlflow.set_tag("model_type", "LightGBM")
            mlflow.set_tag("task", "Sentiment Analysis")
            mlflow.set_tag("dataset", "YouTube Comments")
            mlflow.set_tag("stage", "evaluation")
            mlflow.set_tag("drift_detected", str(drift_detected))
            mlflow.set_tag("quality_passed", str(quality_passed))
            
            # Save comprehensive experiment info
            model_path = f"runs:/{run_id}/model"
            experiment_info = save_experiment_info(
                run_id=run_id,
                model_path=model_path,
                metrics=metrics,
                params=model_params,
                drift_summary=drift_summary,
                quality_summary=quality_summary
            )


            
            # Summary
            logger.info(f"\n{'='*60}")
            logger.info("EVALUATION COMPLETED SUCCESSFULLY")
            logger.info(f"{'='*60}")
            logger.info(f"Run ID: {run_id}")
            logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
            logger.info(f"F1 Score (weighted): {metrics['f1_score_weighted']:.4f}")
            logger.info(f"Drift Detected: {drift_detected}")
            logger.info(f"Quality Check: {'PASSED' if quality_passed else 'FAILED'}")
            logger.info(f"{'='*60}\n")
            
            logger.info("📊 Reports generated:")
            logger.info(f"  - Confusion Matrix: {cm_path}")
            logger.info(f"  - Metrics Report: {metrics_report_path}")
            logger.info(f"  - Drift Reports: reports/drift/")
            logger.info(f"  - Quality Reports: reports/data_quality/")
            logger.info(f"\n✅ All metrics and artifacts logged to MLflow!")
            
            # Alert if issues detected
            if drift_detected:
                logger.warning("\n⚠️  DATA DRIFT DETECTED! Consider retraining the model.")
            if not quality_passed:
                logger.warning("\n⚠️  DATA QUALITY ISSUES DETECTED! Review the quality report.")
            
            return 0
            
    except Exception as e:
        logger.error(f"\n❌ Evaluation failed: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
