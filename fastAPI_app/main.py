# fastAPI_app/main.py
"""
Production FastAPI - Using existing preprocessing and model loading.
"""

import matplotlib

matplotlib.use("Agg")

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import sys
from pathlib import Path
import logging
from dotenv import load_dotenv
import os
import mlflow
from mlflow.tracking import MlflowClient
import pickle

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

# ✅ Import from YOUR existing preprocessing
from src.data.data_preprocessing import preprocess_comment

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(title="YouTube Sentiment Analysis API", version="2.0.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
model = None
vectorizer = None
model_metadata = {}

# Settings from .env
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
MODEL_NAME = os.getenv("MODEL_REGISTRY_NAME", "yt_chrome_plugin_model")
MODEL_ALIAS = os.getenv("MODEL_ALIAS")  # Optional
MODEL_STAGE = os.getenv("MODEL_STAGE", "Production")  # Default to Production


# ==================== MODEL LOADING ====================


def load_model_from_mlflow(model_name: str, alias: str = "champion"):
    """Load model from MLflow registry by alias."""
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        client = MlflowClient()

        # Load by alias
        model_version = client.get_model_version_by_alias(model_name, alias)
        version = model_version.version
        run_id = model_version.run_id

        logger.info(f"Loading model v{version} (alias: {alias})...")

        # Load model
        model_uri = f"models:/{model_name}/{version}"
        loaded_model = mlflow.sklearn.load_model(model_uri)

        # Load vectorizer from artifacts
        vectorizer_path = mlflow.artifacts.download_artifacts(
            run_id=run_id, artifact_path="vectorizer/tfidf_vectorizer.pkl"
        )

        with open(vectorizer_path, "rb") as f:
            loaded_vectorizer = pickle.load(f)

        metadata = {"version": version, "run_id": run_id, "alias": alias}

        logger.info(f"✓ Model v{version} loaded from MLflow")
        return loaded_model, loaded_vectorizer, metadata

    except Exception as e:
        logger.error(f"Failed to load from MLflow by alias: {e}")
        raise


def load_model_from_mlflow_by_stage(model_name: str, stage: str = "Production"):
    """Load model from MLflow registry by stage."""
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        client = MlflowClient()

        # Get latest version in stage
        versions = client.get_latest_versions(model_name, stages=[stage])
        
        if not versions:
            raise Exception(f"No model found in stage: {stage}")
        
        model_version = versions[0]
        version = model_version.version
        run_id = model_version.run_id

        logger.info(f"Loading model v{version} (stage: {stage})...")

        # Load model
        model_uri = f"models:/{model_name}/{version}"
        loaded_model = mlflow.sklearn.load_model(model_uri)

        # Load vectorizer from artifacts
        vectorizer_path = mlflow.artifacts.download_artifacts(
            run_id=run_id, artifact_path="vectorizer/tfidf_vectorizer.pkl"
        )

        with open(vectorizer_path, "rb") as f:
            loaded_vectorizer = pickle.load(f)

        metadata = {"version": version, "run_id": run_id, "stage": stage}

        logger.info(f"✓ Model v{version} loaded from MLflow (stage: {stage})")
        return loaded_model, loaded_vectorizer, metadata

    except Exception as e:
        logger.error(f"Failed to load from MLflow by stage: {e}")
        raise


def load_model_from_local():
    """Fallback: Load from local pickle files."""
    try:
        logger.warning("Loading from local files (fallback)")

        model_path = project_root / "lgbm_model.pkl"
        vectorizer_path = project_root / "tfidf_vectorizer.pkl"

        with open(model_path, "rb") as f:
            loaded_model = pickle.load(f)

        with open(vectorizer_path, "rb") as f:
            loaded_vectorizer = pickle.load(f)

        metadata = {"source": "local_fallback"}

        logger.info("✓ Model loaded from local files")
        return loaded_model, loaded_vectorizer, metadata

    except Exception as e:
        logger.error(f"Failed to load from local: {e}")
        raise


# ==================== STARTUP ====================


@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    global model, vectorizer, model_metadata

    logger.info("Starting application...")

    try:
        # Try MLflow first
        if MLFLOW_TRACKING_URI:
            try:
                if MODEL_ALIAS:
                    # Try alias first if provided
                    logger.info(f"Attempting to load model by alias: {MODEL_ALIAS}")
                    model, vectorizer, model_metadata = load_model_from_mlflow(
                        MODEL_NAME, MODEL_ALIAS
                    )
                else:
                    # Use stage if no alias
                    logger.info(f"Attempting to load model by stage: {MODEL_STAGE}")
                    model, vectorizer, model_metadata = load_model_from_mlflow_by_stage(
                        MODEL_NAME, MODEL_STAGE
                    )
            except Exception as e:
                logger.warning(f"MLflow load failed: {e}. Trying local...")
                model, vectorizer, model_metadata = load_model_from_local()
        else:
            logger.info("No MLflow URI provided, loading from local files")
            model, vectorizer, model_metadata = load_model_from_local()

        logger.info("✅ Model loaded successfully!")
        logger.info(f"Model metadata: {model_metadata}")

    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        logger.error("Application will start but predictions will fail!")


# ==================== PYDANTIC MODELS ====================


class CommentItem(BaseModel):
    text: str
    timestamp: str


class PredictRequest(BaseModel):
    comments: List[str]


class PredictWithTimestampsRequest(BaseModel):
    comments: List[CommentItem]


class PredictResponse(BaseModel):
    comment: str
    sentiment: str
    confidence: Optional[float] = None


# ==================== ENDPOINTS ====================


@app.get("/")
async def home():
    """Root endpoint."""
    return {"message": "YouTube Sentiment Analysis API", "status": "running"}


@app.get("/health")
async def health_check():
    """Health check."""
    return {
        "status": "healthy" if model else "unhealthy",
        "model_loaded": model is not None,
        "model_version": model_metadata.get("version", "unknown"),
    }


@app.get("/get_youtube_api_key")
async def get_youtube_api_key():
    """Get YouTube API key."""
    api_key = os.getenv("YOUTUBE_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="API key not configured")
    return {"api_key": api_key}


@app.post("/predict", response_model=List[PredictResponse])
async def predict(request: PredictRequest):
    """
    Predict sentiment using YOUR existing preprocessing.
    """
    if model is None or vectorizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    comments = request.comments

    if not comments:
        raise HTTPException(status_code=400, detail="No comments provided")

    try:
        # ✅ Use YOUR preprocessing function
        preprocessed = [preprocess_comment(c) for c in comments]

        # Vectorize
        transformed = vectorizer.transform(preprocessed)

        # Predict
        predictions = model.predict(transformed).tolist()

        # Confidence (optional)
        try:
            proba = model.predict_proba(transformed)
            confidences = [float(max(p)) for p in proba]
        except:
            confidences = [None] * len(predictions)

        # Response
        response = [
            {"comment": comment, "sentiment": str(pred), "confidence": conf}
            for comment, pred, conf in zip(comments, predictions, confidences)
        ]

        return response

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict_with_timestamps")
async def predict_with_timestamps(request: PredictWithTimestampsRequest):
    """Predict with timestamps."""
    if model is None or vectorizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    comments_data = request.comments

    if not comments_data:
        raise HTTPException(status_code=400, detail="No comments provided")

    try:
        comments = [item.text for item in comments_data]
        timestamps = [item.timestamp for item in comments_data]

        # ✅ Use YOUR preprocessing
        preprocessed = [preprocess_comment(c) for c in comments]

        # Vectorize & predict
        transformed = vectorizer.transform(preprocessed)
        predictions = model.predict(transformed).tolist()

        # Confidence
        try:
            proba = model.predict_proba(transformed)
            confidences = [float(max(p)) for p in proba]
        except:
            confidences = [None] * len(predictions)

        # Response
        response = [
            {
                "comment": comment,
                "sentiment": str(pred),
                "timestamp": timestamp,
                "confidence": conf,
            }
            for comment, pred, timestamp, conf in zip(
                comments, predictions, timestamps, confidences
            )
        ]

        return response

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("fastAPI_app.main:app", host="0.0.0.0", port=8000, reload=True)