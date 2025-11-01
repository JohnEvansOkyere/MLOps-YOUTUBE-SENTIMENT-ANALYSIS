# model/register_model.py
"""
Model Registration Script for Production ML Pipeline
Registers trained models to MLflow Model Registry with validation and S3 backup
"""

import json
import mlflow
import logging
import os
import sys
from datetime import datetime
from typing import Dict, Optional, Tuple
from dotenv import load_dotenv
from pathlib import Path
import time
import requests
from mlflow.tracking import MlflowClient
from mlflow.exceptions import RestException

# Load environment variables
load_dotenv()

# Configuration from environment variables
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
MODEL_REGISTRY_NAME = os.getenv("MODEL_REGISTRY_NAME", "yt_chrome_plugin_model")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
GIT_COMMIT_HASH = os.getenv("GIT_COMMIT_HASH", "unknown")
AUTO_PROMOTE_TO_PRODUCTION = (
    os.getenv("AUTO_PROMOTE_TO_PRODUCTION", "false").lower() == "true"
)

# Metric thresholds for production promotion
MIN_ACCURACY_THRESHOLD = float(os.getenv("MIN_ACCURACY_THRESHOLD", "0.75"))
MIN_F1_THRESHOLD = float(os.getenv("MIN_F1_THRESHOLD", "0.70"))
PERFORMANCE_IMPROVEMENT_THRESHOLD = float(
    os.getenv("PERFORMANCE_IMPROVEMENT_THRESHOLD", "0.02")
)

# Retry configuration
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
RETRY_DELAY = int(os.getenv("RETRY_DELAY", "5"))


# ==================== LOGGING SETUP ====================
def setup_logging() -> logging.Logger:
    """Configure logging with both console and file handlers."""
    logger = logging.getLogger("model_registration")
    logger.setLevel(logging.DEBUG)

    # Remove existing handlers
    logger.handlers = []

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # File handler for errors
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    file_handler = logging.FileHandler(log_dir / "model_registration_errors.log")
    file_handler.setLevel(logging.ERROR)

    # Detailed file handler for all logs
    detailed_handler = logging.FileHandler(log_dir / "model_registration.log")
    detailed_handler.setLevel(logging.DEBUG)

    # Formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    detailed_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    logger.addHandler(detailed_handler)

    return logger


logger = setup_logging()


# ==================== VALIDATION FUNCTIONS ====================
def validate_environment_variables() -> bool:
    """Validate that all required environment variables are set."""
    required_vars = {
        "MLFLOW_TRACKING_URI": MLFLOW_TRACKING_URI,
        "MODEL_REGISTRY_NAME": MODEL_REGISTRY_NAME,
    }

    missing_vars = [var for var, value in required_vars.items() if not value]

    if missing_vars:
        logger.error(
            f"Missing required environment variables: {', '.join(missing_vars)}"
        )
        return False

    logger.info("✓ All required environment variables are set")
    return True


def check_mlflow_connection() -> bool:
    """Check if MLflow server is reachable."""
    try:
        response = requests.get(f"{MLFLOW_TRACKING_URI}/health", timeout=10)
        if response.status_code == 200:
            logger.info("✓ Successfully connected to MLflow server")
            return True
        else:
            logger.error(f"MLflow server returned status code: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to connect to MLflow server: {e}")
        return False


def load_model_info(file_path: str = "experiment_info.json") -> Dict:
    """Load and validate model info from JSON file."""
    try:
        if not Path(file_path).exists():
            raise FileNotFoundError(f"Model info file not found: {file_path}")

        with open(file_path, "r") as file:
            model_info = json.load(file)

        # Validate required fields
        required_fields = ["run_id", "model_path"]
        missing_fields = [field for field in required_fields if field not in model_info]

        if missing_fields:
            raise ValueError(
                f"Missing required fields in model_info: {', '.join(missing_fields)}"
            )

        logger.info(f"✓ Model info loaded from {file_path}")
        logger.debug(f"Run ID: {model_info['run_id']}")
        logger.debug(f"Model Path: {model_info['model_path']}")

        return model_info

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in {file_path}: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error loading model info: {e}")
        raise


def validate_model_artifact(client: MlflowClient, run_id: str, model_path: str) -> bool:
    """Validate that the model artifact exists and is loadable."""
    try:
        model_uri = f"runs:/{run_id}/{model_path}"

        # Try to load the model to ensure it's valid
        logger.info(f"Validating model artifact: {model_uri}")
        model = mlflow.pyfunc.load_model(model_uri)

        logger.info("✓ Model artifact validated successfully")
        return True

    except Exception as e:
        logger.error(f"Model validation failed: {e}")
        return False


# ==================== MODEL COMPARISON ====================
def get_current_production_model(
    client: MlflowClient, model_name: str
) -> Optional[Dict]:
    """Get current production model version and its metrics."""
    try:
        # Get production versions
        prod_versions = client.get_latest_versions(model_name, stages=["Production"])

        if not prod_versions:
            logger.info("No current production model found")
            return None

        prod_version = prod_versions[0]
        run = client.get_run(prod_version.run_id)

        return {
            "version": prod_version.version,
            "run_id": prod_version.run_id,
            "metrics": run.data.metrics,
        }

    except RestException as e:
        if "RESOURCE_DOES_NOT_EXIST" in str(e):
            logger.info(
                f"Model '{model_name}' not found in registry. This will be the first version."
            )
            return None
        raise


def compare_model_performance(
    new_metrics: Dict, prod_metrics: Optional[Dict]
) -> Tuple[bool, str]:
    """
    Compare new model metrics with production model.
    Returns (should_promote, reason)
    """
    if prod_metrics is None:
        return (
            True,
            "No production model exists - auto-promoting first model to staging",
        )

    # Extract key metrics
    new_accuracy = new_metrics.get("accuracy", 0)
    new_f1 = new_metrics.get("f1_score", 0)

    prod_accuracy = prod_metrics.get("accuracy", 0)
    prod_f1 = prod_metrics.get("f1_score", 0)

    # Check minimum thresholds
    if new_accuracy < MIN_ACCURACY_THRESHOLD:
        return (
            False,
            f"Accuracy {new_accuracy:.4f} below threshold {MIN_ACCURACY_THRESHOLD}",
        )

    if new_f1 < MIN_F1_THRESHOLD:
        return False, f"F1 Score {new_f1:.4f} below threshold {MIN_F1_THRESHOLD}"

    # Compare with production
    accuracy_improvement = new_accuracy - prod_accuracy
    f1_improvement = new_f1 - prod_f1

    if (
        accuracy_improvement >= PERFORMANCE_IMPROVEMENT_THRESHOLD
        or f1_improvement >= PERFORMANCE_IMPROVEMENT_THRESHOLD
    ):
        return (
            True,
            f"Performance improved (Acc: +{accuracy_improvement:.4f}, F1: +{f1_improvement:.4f})",
        )

    if accuracy_improvement >= 0 and f1_improvement >= 0:
        return True, "Performance maintained or slightly improved"

    return (
        False,
        f"Performance degraded (Acc: {accuracy_improvement:.4f}, F1: {f1_improvement:.4f})",
    )


# ==================== MODEL REGISTRATION ====================
def register_model_with_retry(
    model_name: str, model_info: Dict, client: MlflowClient
) -> mlflow.entities.model_registry.ModelVersion:
    """Register model with retry logic."""
    model_uri = f"runs:/{model_info['run_id']}/{model_info['model_path']}"

    for attempt in range(MAX_RETRIES):
        try:
            logger.info(
                f"Attempting to register model (attempt {attempt + 1}/{MAX_RETRIES})"
            )
            model_version = mlflow.register_model(model_uri, model_name)
            logger.info(
                f"✓ Model registered successfully as version {model_version.version}"
            )
            return model_version

        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                logger.warning(f"Registration failed, retrying in {RETRY_DELAY}s: {e}")
                time.sleep(RETRY_DELAY)
            else:
                logger.error(
                    f"Failed to register model after {MAX_RETRIES} attempts: {e}"
                )
                raise


def add_model_metadata(
    client: MlflowClient, model_name: str, version: str, model_info: Dict, metrics: Dict
):
    """Add comprehensive metadata to registered model."""
    try:
        # Set model version description
        description = f"""
        YouTube Sentiment Analysis Model
        Registered: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        Environment: {ENVIRONMENT}
        Git Commit: {GIT_COMMIT_HASH}
        
        Performance Metrics:
        - Accuracy: {metrics.get('accuracy', 'N/A')}
        - F1 Score: {metrics.get('f1_score', 'N/A')}
        - Precision: {metrics.get('precision', 'N/A')}
        - Recall: {metrics.get('recall', 'N/A')}
        """

        client.update_model_version(
            name=model_name, version=version, description=description
        )

        # Set tags
        tags = {
            "git_commit": GIT_COMMIT_HASH,
            "environment": ENVIRONMENT,
            "registration_date": datetime.now().isoformat(),
            "model_type": model_info.get("model_type", "sentiment_classifier"),
            "dataset_version": model_info.get("dataset_version", "unknown"),
            "framework": "tensorflow",  # Update based on your framework
            "accuracy": str(metrics.get("accuracy", "N/A")),
            "f1_score": str(metrics.get("f1_score", "N/A")),
        }

        for key, value in tags.items():
            client.set_model_version_tag(model_name, version, key, value)

        logger.info("✓ Model metadata added successfully")

    except Exception as e:
        logger.warning(f"Failed to add model metadata: {e}")


def transition_model_stage(
    client: MlflowClient,
    model_name: str,
    version: str,
    target_stage: str,
    archive_existing: bool = True,
):
    """Transition model using aliases (new MLflow approach)."""
    try:
        # Use aliases instead of stages
        alias_map = {"Staging": "challenger", "Production": "champion", "None": None}

        alias = alias_map.get(target_stage)
        if alias:
            client.set_registered_model_alias(model_name, alias, version)
            logger.info(f"✓ Model version {version} set to alias '{alias}'")
        else:
            logger.info(f"✓ Model version {version} registered without alias")

    except Exception as e:
        logger.error(f"Failed to set model alias: {e}")
        raise


def set_model_alias(client: MlflowClient, model_name: str, version: str, alias: str):
    """Set alias for model version (e.g., 'champion', 'challenger')."""
    try:
        client.set_registered_model_alias(model_name, alias, version)
        logger.info(f"✓ Alias '{alias}' set for model version {version}")
    except Exception as e:
        logger.warning(f"Failed to set alias '{alias}': {e}")


# ==================== MAIN REGISTRATION FLOW ====================
def register_model_to_mlflow(
    model_info: Dict, model_name: str = MODEL_REGISTRY_NAME
) -> Dict:
    """
    Main function to register model to MLflow with validation and metadata.
    Returns registration summary.
    """
    summary = {
        "success": False,
        "model_name": model_name,
        "version": None,
        "stage": None,
        "timestamp": datetime.now().isoformat(),
    }

    try:
        # Set MLflow tracking URI
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        client = MlflowClient()

        logger.info("=" * 60)
        logger.info("STARTING MODEL REGISTRATION PROCESS")
        logger.info("=" * 60)

        # 1. Validate environment
        logger.info("\n[1/8] Validating environment...")
        if not validate_environment_variables():
            raise ValueError("Environment validation failed")

        # 2. Check MLflow connection
        logger.info("\n[2/8] Checking MLflow connection...")
        if not check_mlflow_connection():
            raise ConnectionError("Cannot connect to MLflow server")

        # 3. Validate model artifact
        logger.info("\n[3/8] Validating model artifact...")
        if not validate_model_artifact(
            client, model_info["run_id"], model_info["model_path"]
        ):
            raise ValueError("Model artifact validation failed")

        # 4. Get metrics from the run
        logger.info("\n[4/8] Fetching model metrics...")
        run = client.get_run(model_info["run_id"])
        new_metrics = run.data.metrics
        logger.info(f"New model metrics: {json.dumps(new_metrics, indent=2)}")

        # 5. Get current production model
        logger.info("\n[5/8] Checking current production model...")
        prod_model = get_current_production_model(client, model_name)

        # 6. Compare performance
        logger.info("\n[6/8] Comparing model performance...")
        should_promote, reason = compare_model_performance(
            new_metrics, prod_model["metrics"] if prod_model else None
        )
        logger.info(f"Promotion decision: {should_promote} - {reason}")

        # 7. Register model
        logger.info("\n[7/8] Registering model...")
        model_version = register_model_with_retry(model_name, model_info, client)
        summary["version"] = model_version.version

        # Add metadata
        add_model_metadata(
            client, model_name, model_version.version, model_info, new_metrics
        )

        # 8. Stage transition
        logger.info("\n[8/8] Managing model stages...")

        if should_promote:
            # Transition to Staging first
            transition_model_stage(client, model_name, model_version.version, "Staging")
            summary["stage"] = "Staging"
            set_model_alias(client, model_name, model_version.version, "challenger")

            # Auto-promote to Production if enabled and metrics are good
            if AUTO_PROMOTE_TO_PRODUCTION:
                logger.info("Auto-promotion to Production is enabled")
                transition_model_stage(
                    client, model_name, model_version.version, "Production"
                )
                summary["stage"] = "Production"
                set_model_alias(client, model_name, model_version.version, "champion")
                logger.info("✓ Model promoted to Production")
        else:
            # Keep in None stage
            logger.warning(f"Model not promoted: {reason}")
            summary["stage"] = "None"
            summary["promotion_blocked_reason"] = reason

        summary["success"] = True
        summary["metrics"] = new_metrics
        summary["promotion_reason"] = reason

        logger.info("\n" + "=" * 60)
        logger.info("MODEL REGISTRATION COMPLETED SUCCESSFULLY")
        logger.info("=" * 60)
        logger.info(f"Model Name: {model_name}")
        logger.info(f"Version: {model_version.version}")
        logger.info(f"Stage: {summary['stage']}")
        logger.info(f"Run ID: {model_info['run_id']}")
        logger.info("=" * 60)

        return summary

    except Exception as e:
        logger.error(f"\n{'=' * 60}")
        logger.error("MODEL REGISTRATION FAILED")
        logger.error(f"{'=' * 60}")
        logger.error(f"Error: {str(e)}", exc_info=True)
        summary["error"] = str(e)
        raise


# ==================== MAIN ====================
def main():
    """Main entry point for model registration script."""
    exit_code = 0

    try:
        # Load model info
        model_info_path = os.getenv("MODEL_INFO_PATH", "experiment_info.json")
        model_info = load_model_info(model_info_path)

        # Register model
        summary = register_model_to_mlflow(model_info)

        # Save summary for CI/CD
        summary_path = Path("model_registration_summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info(f"\nRegistration summary saved to {summary_path}")

        # Print summary for CI/CD pipeline
        print("\n" + "=" * 60)
        print("REGISTRATION SUMMARY (JSON)")
        print("=" * 60)
        print(json.dumps(summary, indent=2))
        print("=" * 60)

    except Exception as e:
        logger.error(f"Registration process failed: {e}")
        exit_code = 1

        # Create failure summary
        failure_summary = {
            "success": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
        }

        with open("model_registration_summary.json", "w") as f:
            json.dump(failure_summary, f, indent=2)

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
