# model/model_building.py
"""
Model Training Script for YouTube Sentiment Analysis
Trains LightGBM classifier with MLflow tracking and prepares for registration
"""

import numpy as np
import pandas as pd
import os
import pickle
import yaml
import logging
import json
from pathlib import Path
from datetime import datetime
from typing import Tuple, Dict
import lightgbm as lgb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# MLflow Configuration
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "yt_sentiment_analysis")
MODEL_NAME = os.getenv("MODEL_REGISTRY_NAME", "yt_chrome_plugin_model")
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
GIT_COMMIT_HASH = os.getenv("GIT_COMMIT_HASH", "unknown")


# ==================== LOGGING SETUP ====================
def setup_logging() -> logging.Logger:
    """Configure logging with both console and file handlers."""
    logger = logging.getLogger("model_building")
    logger.setLevel(logging.DEBUG)

    # Remove existing handlers
    logger.handlers = []

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Create logs directory
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    # File handler for errors
    file_handler = logging.FileHandler(log_dir / "model_building_errors.log")
    file_handler.setLevel(logging.ERROR)

    # Detailed file handler
    detailed_handler = logging.FileHandler(log_dir / "model_building.log")
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


# ==================== UTILITY FUNCTIONS ====================
def get_root_directory() -> Path:
    """Get the root directory (two levels up from this script's location)."""
    current_dir = Path(__file__).resolve().parent
    return current_dir.parent.parent


def load_params(params_path: str) -> Dict:
    """Load parameters from a YAML file."""
    try:
        with open(params_path, "r") as file:
            params = yaml.safe_load(file)
        logger.info(f"✓ Parameters loaded from {params_path}")
        return params
    except FileNotFoundError:
        logger.error(f"Parameters file not found: {params_path}")
        raise
    except yaml.YAMLError as e:
        logger.error(f"YAML parsing error: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error loading parameters: {e}")
        raise


def load_data(file_path: str) -> pd.DataFrame:
    """Load data from a CSV file with validation."""
    try:
        if not Path(file_path).exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")

        df = pd.read_csv(file_path)

        # Validate required columns
        required_columns = ["clean_comment", "category"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        # Handle missing values
        original_shape = df.shape
        df.fillna("", inplace=True)

        logger.info(f"✓ Data loaded from {file_path}")
        logger.info(f"  Shape: {df.shape}")
        logger.info(f"  Columns: {list(df.columns)}")

        if df.isna().sum().sum() > 0:
            logger.warning(f"  Filled {df.isna().sum().sum()} NaN values")

        return df

    except pd.errors.ParserError as e:
        logger.error(f"Failed to parse CSV file: {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise


# ==================== DATA VALIDATION ====================
def validate_data(df: pd.DataFrame) -> Dict:
    """Validate training data and return statistics."""
    stats = {
        "total_samples": len(df),
        "features": list(df.columns),
        "target_distribution": df["category"].value_counts().to_dict(),
        "missing_values": df.isna().sum().to_dict(),
        "empty_comments": (df["clean_comment"] == "").sum(),
    }

    logger.info("Data Validation:")
    logger.info(f"  Total samples: {stats['total_samples']}")
    logger.info(f"  Target distribution: {stats['target_distribution']}")
    logger.info(f"  Empty comments: {stats['empty_comments']}")

    # Check for class imbalance
    min_class_size = min(stats["target_distribution"].values())
    max_class_size = max(stats["target_distribution"].values())
    imbalance_ratio = (
        max_class_size / min_class_size if min_class_size > 0 else float("inf")
    )

    if imbalance_ratio > 3:
        logger.warning(
            f"  ⚠ Significant class imbalance detected (ratio: {imbalance_ratio:.2f})"
        )

    return stats


# ==================== FEATURE ENGINEERING ====================
def apply_tfidf(
    train_data: pd.DataFrame,
    max_features: int,
    ngram_range: Tuple[int, int],
    root_dir: Path,
) -> Tuple[np.ndarray, np.ndarray, TfidfVectorizer]:
    """
    Apply TF-IDF vectorization with ngrams to the training data.
    Returns transformed features, labels, and fitted vectorizer.
    """
    try:
        logger.info(f"\n{'='*60}")
        logger.info("FEATURE ENGINEERING - TF-IDF")
        logger.info(f"{'='*60}")

        # Initialize vectorizer
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=2,  # Ignore terms that appear in less than 2 documents
            max_df=0.95,  # Ignore terms that appear in more than 95% of documents
            sublinear_tf=True,  # Use sublinear term frequency scaling
        )

        # Extract features and labels
        X_train = train_data["clean_comment"].values
        y_train = train_data["category"].values

        logger.info(f"Input shape: {X_train.shape}")
        logger.info(f"TF-IDF parameters:")
        logger.info(f"  max_features: {max_features}")
        logger.info(f"  ngram_range: {ngram_range}")

        # Fit and transform
        X_train_tfidf = vectorizer.fit_transform(X_train)

        logger.info(f"✓ TF-IDF transformation complete")
        logger.info(f"  Output shape: {X_train_tfidf.shape}")
        logger.info(f"  Vocabulary size: {len(vectorizer.vocabulary_)}")
        logger.info(f"  Non-zero elements: {X_train_tfidf.nnz}")
        logger.info(
            f"  Sparsity: {100 * (1 - X_train_tfidf.nnz / (X_train_tfidf.shape[0] * X_train_tfidf.shape[1])):.2f}%"
        )

        # Save vectorizer
        vectorizer_path = root_dir / "tfidf_vectorizer.pkl"
        with open(vectorizer_path, "wb") as f:
            pickle.dump(vectorizer, f)
        logger.info(f"✓ Vectorizer saved to {vectorizer_path}")

        return X_train_tfidf, y_train, vectorizer

    except Exception as e:
        logger.error(f"Error during TF-IDF transformation: {e}")
        raise


# ==================== MODEL TRAINING ====================
def train_lgbm(
    X_train: np.ndarray, y_train: np.ndarray, params: Dict
) -> lgb.LGBMClassifier:
    """Train a LightGBM classifier with specified parameters."""
    try:
        logger.info(f"\n{'='*60}")
        logger.info("MODEL TRAINING - LightGBM")
        logger.info(f"{'='*60}")

        # Extract parameters
        learning_rate = params["learning_rate"]
        max_depth = params["max_depth"]
        n_estimators = params["n_estimators"]

        logger.info(f"Model parameters:")
        logger.info(f"  learning_rate: {learning_rate}")
        logger.info(f"  max_depth: {max_depth}")
        logger.info(f"  n_estimators: {n_estimators}")

        # Initialize model
        model = lgb.LGBMClassifier(
            objective="multiclass",
            num_class=3,
            metric="multi_logloss",
            is_unbalance=True,
            class_weight="balanced",
            reg_alpha=0.1,  # L1 regularization
            reg_lambda=0.1,  # L2 regularization
            learning_rate=learning_rate,
            max_depth=max_depth,
            n_estimators=n_estimators,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )

        # Train model
        logger.info("Training model...")
        model.fit(X_train, y_train)

        logger.info("✓ Model training completed")

        return model

    except Exception as e:
        logger.error(f"Error during model training: {e}")
        raise


# ==================== MODEL EVALUATION ====================
def evaluate_model(
    model: lgb.LGBMClassifier, X: np.ndarray, y: np.ndarray, dataset_name: str = "train"
) -> Dict:
    """Evaluate model and return metrics."""
    try:
        logger.info(f"\n{'='*60}")
        logger.info(f"MODEL EVALUATION - {dataset_name.upper()}")
        logger.info(f"{'='*60}")

        # Make predictions
        y_pred = model.predict(X)
        y_pred_proba = model.predict_proba(X)

        # Calculate metrics
        metrics = {
            "accuracy": accuracy_score(y, y_pred),
            "precision": precision_score(y, y_pred, average="weighted"),
            "recall": recall_score(y, y_pred, average="weighted"),
            "f1_score": f1_score(y, y_pred, average="weighted"),
            "precision_macro": precision_score(y, y_pred, average="macro"),
            "recall_macro": recall_score(y, y_pred, average="macro"),
            "f1_score_macro": f1_score(y, y_pred, average="macro"),
        }

        # Log metrics
        logger.info("Metrics:")
        logger.info(f"  Accuracy:  {metrics['accuracy']:.4f}")
        logger.info(f"  Precision: {metrics['precision']:.4f}")
        logger.info(f"  Recall:    {metrics['recall']:.4f}")
        logger.info(f"  F1 Score:  {metrics['f1_score']:.4f}")

        # Classification report
        logger.info("\nClassification Report:")
        report = classification_report(y, y_pred)
        logger.info(f"\n{report}")

        # Confusion matrix
        cm = confusion_matrix(y, y_pred)
        logger.info("\nConfusion Matrix:")
        logger.info(f"\n{cm}")

        return metrics

    except Exception as e:
        logger.error(f"Error during model evaluation: {e}")
        raise


# ==================== MLFLOW INTEGRATION ====================
def save_experiment_info(
    run_id: str, model_path: str, metrics: Dict, params: Dict, root_dir: Path
):
    """Save experiment information for model registration."""
    try:
        experiment_info = {
            "run_id": run_id,
            "model_path": model_path,
            "model_type": "lightgbm",
            "framework": "lightgbm",
            "metrics": metrics,
            "parameters": params,
            "timestamp": datetime.now().isoformat(),
            "git_commit": GIT_COMMIT_HASH,
            "environment": ENVIRONMENT,
            "dataset_version": "v1.0",  # Update this based on your versioning
        }

        # Save to root directory
        info_path = root_dir / "experiment_info.json"
        with open(info_path, "w") as f:
            json.dump(experiment_info, f, indent=2)

        logger.info(f"✓ Experiment info saved to {info_path}")

        return experiment_info

    except Exception as e:
        logger.error(f"Error saving experiment info: {e}")
        raise


def train_with_mlflow(root_dir: Path, params: Dict):
    """Main training function with MLflow tracking."""

    # Set MLflow tracking URI
    if MLFLOW_TRACKING_URI:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        logger.info(f"✓ MLflow tracking URI: {MLFLOW_TRACKING_URI}")

    # Set experiment
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    logger.info(f"✓ MLflow experiment: {MLFLOW_EXPERIMENT_NAME}")

    # Start MLflow run
    with mlflow.start_run(
        run_name=f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    ) as run:
        run_id = run.info.run_id
        logger.info(f"\n{'='*60}")
        logger.info(f"MLFLOW RUN STARTED: {run_id}")
        logger.info(f"{'='*60}\n")

        try:
            # Extract parameters
            model_params = params["model_building"]
            max_features = model_params["max_features"]
            ngram_range = tuple(model_params["ngram_range"])
            learning_rate = model_params["learning_rate"]
            max_depth = model_params["max_depth"]
            n_estimators = model_params["n_estimators"]

            # Log parameters to MLflow
            mlflow.log_param("max_features", max_features)
            mlflow.log_param("ngram_range", str(ngram_range))
            mlflow.log_param("learning_rate", learning_rate)
            mlflow.log_param("max_depth", max_depth)
            mlflow.log_param("n_estimators", n_estimators)
            mlflow.log_param("model_type", "lightgbm")
            mlflow.log_param("objective", "multiclass")
            mlflow.log_param("num_classes", 3)
            mlflow.log_param("git_commit", GIT_COMMIT_HASH)
            mlflow.log_param("environment", ENVIRONMENT)

            # Load training data
            train_data_path = root_dir / "data" / "interim" / "train_processed.csv"
            train_data = load_data(str(train_data_path))

            # Validate data
            data_stats = validate_data(train_data)
            mlflow.log_param("train_samples", data_stats["total_samples"])
            mlflow.log_param(
                "target_distribution", str(data_stats["target_distribution"])
            )

            # Apply TF-IDF
            X_train_tfidf, y_train, vectorizer = apply_tfidf(
                train_data, max_features, ngram_range, root_dir
            )

            # Log feature engineering stats
            mlflow.log_param("vocabulary_size", len(vectorizer.vocabulary_))
            mlflow.log_param("feature_dim", X_train_tfidf.shape[1])

            # Train model
            model = train_lgbm(X_train_tfidf, y_train, model_params)

            # Evaluate model
            metrics = evaluate_model(model, X_train_tfidf, y_train, "train")

            # Log metrics to MLflow
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)

            # Create input example for model signature
            # Take a few samples for the signature
            sample_indices = np.random.choice(
                len(train_data), size=min(5, len(train_data)), replace=False
            )
            input_example = train_data.iloc[sample_indices]["clean_comment"].values

            # Transform input example through vectorizer
            input_example_tfidf = vectorizer.transform(input_example)

            # Infer model signature
            signature = infer_signature(
                input_example_tfidf, model.predict(input_example_tfidf)
            )

            # Log model with MLflow
            model_path = "model"
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path=model_path,
                signature=signature,
                input_example=input_example_tfidf[
                    :2
                ].toarray(),  # Log first 2 samples as example
                registered_model_name=None,  # We'll register separately
            )
            logger.info(f"✓ Model logged to MLflow at path: {model_path}")

            # Log vectorizer as artifact
            vectorizer_path = root_dir / "tfidf_vectorizer.pkl"
            mlflow.log_artifact(str(vectorizer_path), artifact_path="vectorizer")
            logger.info("✓ TF-IDF vectorizer logged as artifact")

            # Log parameters file
            params_path = root_dir / "params.yaml"
            if params_path.exists():
                mlflow.log_artifact(str(params_path))

            # Log data statistics as JSON
            stats_path = root_dir / "data_stats.json"

            # Convert numpy/pandas types to native Python types
            def convert_to_serializable(obj):
                """Convert numpy/pandas types to JSON serializable types."""
                if isinstance(obj, dict):
                    return {k: convert_to_serializable(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_to_serializable(item) for item in obj]
                elif hasattr(obj, "item"):  # numpy types
                    return obj.item()
                elif hasattr(obj, "tolist"):  # numpy arrays
                    return obj.tolist()
                else:
                    return obj

            serializable_stats = convert_to_serializable(data_stats)

            with open(stats_path, "w") as f:
                json.dump(serializable_stats, f, indent=2)
            mlflow.log_artifact(str(stats_path))

            # Save model locally (pickle format for backward compatibility)
            model_pkl_path = root_dir / "lgbm_model.pkl"
            with open(model_pkl_path, "wb") as f:
                pickle.dump(model, f)
            logger.info(f"✓ Model saved locally to {model_pkl_path}")

            # Save experiment info for registration
            experiment_info = save_experiment_info(
                run_id, "model", metrics, model_params, root_dir
            )

            # Set tags
            mlflow.set_tag("model_type", "sentiment_classifier")
            mlflow.set_tag("framework", "lightgbm")
            mlflow.set_tag("stage", "development")
            mlflow.set_tag("author", os.getenv("USER", "unknown"))

            logger.info(f"\n{'='*60}")
            logger.info("TRAINING COMPLETED SUCCESSFULLY")
            logger.info(f"{'='*60}")
            logger.info(f"Run ID: {run_id}")
            logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
            logger.info(f"F1 Score: {metrics['f1_score']:.4f}")
            logger.info(f"Model saved to MLflow and locally")
            logger.info(f"{'='*60}\n")

            return run_id, metrics

        except Exception as e:
            logger.error(f"Error during training: {e}", exc_info=True)
            mlflow.log_param("status", "failed")
            mlflow.log_param("error", str(e))
            raise


# ==================== MAIN ====================
def main():
    """Main entry point for model training script."""
    try:
        logger.info(f"\n{'='*60}")
        logger.info("YOUTUBE SENTIMENT ANALYSIS - MODEL TRAINING")
        logger.info(f"{'='*60}")
        logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Environment: {ENVIRONMENT}")
        logger.info(f"Git commit: {GIT_COMMIT_HASH}")
        logger.info(f"{'='*60}\n")

        # Get root directory
        root_dir = get_root_directory()
        logger.info(f"Root directory: {root_dir}")

        # Load parameters
        params_path = root_dir / "params.yaml"
        params = load_params(str(params_path))

        # Train with MLflow
        run_id, metrics = train_with_mlflow(root_dir, params)

        logger.info("\n✅ Training pipeline completed successfully!")
        logger.info(f"Next step: Run model registration script to register this model")
        logger.info(f"Command: python model/register_model.py")

    except Exception as e:
        logger.error(f"\n❌ Training pipeline failed: {e}", exc_info=True)
        print(f"\nError: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)
