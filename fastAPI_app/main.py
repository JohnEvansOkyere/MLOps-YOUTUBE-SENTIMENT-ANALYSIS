"""
FastAPI application for YouTube sentiment analysis.

This application provides endpoints for sentiment prediction, visualization generation,
and trend analysis using a pre-trained LGBM model with TF-IDF vectorization.
Integrates with MLflow and DVC for model management.
"""

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend before importing pyplot

from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import io
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import mlflow
import numpy as np
import re
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from mlflow.tracking import MlflowClient
import matplotlib.dates as mdates
import pickle
import logging
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get base directory (parent of current file's directory)
# If this file is in /app/fastAPI_app/main.py, BASE_DIR will be /app
BASE_DIR = Path(__file__).resolve().parent.parent

# Initialize FastAPI app
app = FastAPI(
    title="YouTube Sentiment Analysis API",
    description="API for analyzing sentiment in YouTube comments using ML models",
    version="1.0.0"
)

# Configure CORS - Enable CORS for all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


# ========== Pydantic Models for Request/Response Validation ==========

class CommentItem(BaseModel):
    """Model for a single comment with timestamp."""
    text: str = Field(..., description="The comment text to analyze")
    timestamp: str = Field(..., description="Timestamp when the comment was posted")


class PredictWithTimestampsRequest(BaseModel):
    """Request model for prediction with timestamps."""
    comments: List[CommentItem] = Field(..., description="List of comments with timestamps")


class PredictRequest(BaseModel):
    """Request model for basic prediction."""
    comments: List[str] = Field(..., description="List of comment texts to analyze")


class PredictResponse(BaseModel):
    """Response model for prediction results."""
    comment: str = Field(..., description="Original comment text")
    sentiment: str = Field(..., description="Predicted sentiment (-1: Negative, 0: Neutral, 1: Positive)")


class PredictWithTimestampsResponse(BaseModel):
    """Response model for prediction with timestamps."""
    comment: str = Field(..., description="Original comment text")
    sentiment: str = Field(..., description="Predicted sentiment (-1: Negative, 0: Neutral, 1: Positive)")
    timestamp: str = Field(..., description="Comment timestamp")


class SentimentCountsRequest(BaseModel):
    """Request model for sentiment counts chart generation."""
    sentiment_counts: Dict[str, int] = Field(..., description="Dictionary with sentiment counts")


class WordCloudRequest(BaseModel):
    """Request model for word cloud generation."""
    comments: List[str] = Field(..., description="List of comments to generate word cloud from")


class TrendGraphItem(BaseModel):
    """Model for sentiment data with timestamp."""
    sentiment: int = Field(..., description="Sentiment value (-1, 0, or 1)")
    timestamp: str = Field(..., description="Timestamp of the comment")


class TrendGraphRequest(BaseModel):
    """Request model for trend graph generation."""
    sentiment_data: List[TrendGraphItem] = Field(..., description="List of sentiment data with timestamps")


# ========== Preprocessing and Model Loading Functions ==========

def preprocess_comment(comment: str) -> str:
    """
    Apply preprocessing transformations to a comment.
    
    This function performs several text preprocessing steps:
    1. Converts text to lowercase
    2. Removes leading/trailing whitespace
    3. Removes newline characters
    4. Removes non-alphanumeric characters (except basic punctuation)
    5. Removes stopwords (keeping sentiment-important words like 'not', 'but')
    6. Lemmatizes words to their root form
    
    Args:
        comment (str): Raw comment text to preprocess
        
    Returns:
        str: Preprocessed comment text ready for vectorization
        
    Example:
        >>> preprocess_comment("I don't like this video!")
        "like video"
    """
    try:
        # Convert to lowercase for consistency
        comment = comment.lower()

        # Remove trailing and leading whitespaces
        comment = comment.strip()

        # Remove newline characters and replace with spaces
        comment = re.sub(r'\n', ' ', comment)

        # Remove non-alphanumeric characters, except basic punctuation (!?.,)
        comment = re.sub(r'[^A-Za-z0-9\s!?.,]', '', comment)

        # Remove stopwords but retain important ones for sentiment analysis
        # Keep negation words that are crucial for sentiment (not, no, but, etc.)
        stop_words = set(stopwords.words('english')) - {'not', 'but', 'however', 'no', 'yet'}
        comment = ' '.join([word for word in comment.split() if word not in stop_words])

        # Lemmatize the words to their base form (e.g., "running" -> "run")
        lemmatizer = WordNetLemmatizer()
        comment = ' '.join([lemmatizer.lemmatize(word) for word in comment.split()])

        return comment
    except Exception as e:
        logger.error(f"Error in preprocessing comment: {e}")
        # Return original comment if preprocessing fails
        return comment


def load_model(model_path: str, vectorizer_path: str) -> tuple:
    """
    Load the trained model and vectorizer from pickle files.
    
    Args:
        model_path (str): Path to the pickled model file (e.g., "./lgbm_model.pkl")
        vectorizer_path (str): Path to the pickled TF-IDF vectorizer file
        
    Returns:
        tuple: (model, vectorizer) - Loaded model and vectorizer objects
        
    Raises:
        Exception: If loading fails
    """
    try:
        # Load the trained LGBM model from pickle file
        logger.info(f"Loading model from: {model_path}")
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        
        # Load the TF-IDF vectorizer from pickle file
        logger.info(f"Loading vectorizer from: {vectorizer_path}")
        with open(vectorizer_path, 'rb') as file:
            vectorizer = pickle.load(file)
      
        logger.info("Model and vectorizer loaded successfully")
        return model, vectorizer
    except FileNotFoundError as e:
        logger.error(f"Model file not found: {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading model or vectorizer: {e}")
        raise


# Commented out MLflow model loading function for reference
# def load_model_and_vectorizer(model_name: str, model_version: str, vectorizer_path: str) -> tuple:
#     """
#     Load model from MLflow model registry and vectorizer from local storage.
#     
#     Args:
#         model_name (str): Name of the model in MLflow registry
#         model_version (str): Version of the model to load
#         vectorizer_path (str): Path to the pickled vectorizer file
#         
#     Returns:
#         tuple: (model, vectorizer) - Loaded model and vectorizer objects
#     """
#     # Set MLflow tracking URI to your server
#     mlflow.set_tracking_uri("http://ec2-54-167-108-249.compute-1.amazonaws.com:5000/")
#     client = MlflowClient()
#     model_uri = f"models:/{model_name}/{model_version}"
#     model = mlflow.pyfunc.load_model(model_uri)
#     with open(vectorizer_path, 'rb') as file:
#         vectorizer = pickle.load(file)
#     return model, vectorizer


# ========== Initialize Model and Vectorizer ==========

# Build absolute paths to model files using BASE_DIR
# Use environment variables if set, otherwise use default paths
model_path = os.getenv("MODEL_PATH", str(BASE_DIR / "lgbm_model.pkl"))
vectorizer_path = os.getenv("VECTORIZER_PATH", str(BASE_DIR / "tfidf_vectorizer.pkl"))

logger.info(f"Base directory: {BASE_DIR}")
logger.info(f"Model path: {model_path}")
logger.info(f"Vectorizer path: {vectorizer_path}")

# Load model and vectorizer at startup
try:
    model, vectorizer = load_model(model_path, vectorizer_path)
except Exception as e:
    logger.error(f"Failed to load model at startup: {e}")
    logger.error("Application will start but predictions will fail!")
    model, vectorizer = None, None

# Alternative: Load from MLflow model registry (uncomment if needed)
# model, vectorizer = load_model_and_vectorizer("my_model", "1", "./tfidf_vectorizer.pkl")


# ========== API Endpoints ==========

@app.get("/", tags=["Home"])
async def home():
    """
    Root endpoint - API welcome message.
    
    Returns:
        dict: Welcome message
    """
    return {"message": "Welcome to our FastAPI sentiment analysis API"}


@app.get("/get_youtube_api_key", tags=["Configuration"])
async def get_youtube_api_key():
    """
    Get YouTube Data API key for Chrome extension.
    
    This endpoint serves the YouTube API key to the Chrome extension.
    The key is stored securely in environment variables and never committed to Git.
    
    Returns:
        dict: YouTube API key
        
    Raises:
        HTTPException: If API key is not configured
        
    Security Note:
        - This endpoint should be protected with authentication in production
        - Consider implementing rate limiting
        - Use CORS to restrict access to your extension only
    """
    youtube_api_key = os.getenv("YOUTUBE_API_KEY")
    
    if not youtube_api_key:
        logger.error("YouTube API key not found in environment variables")
        raise HTTPException(
            status_code=500,
            detail="YouTube API key not configured. Please set YOUTUBE_API_KEY in .env file"
        )
    
    return {"api_key": youtube_api_key}


@app.post(
    "/predict_with_timestamps",
    response_model=List[PredictWithTimestampsResponse],
    tags=["Prediction"]
)
async def predict_with_timestamps(request: PredictWithTimestampsRequest):
    """
    Predict sentiment for comments with timestamps.
    
    This endpoint analyzes a list of comments with timestamps and returns
    sentiment predictions along with the original comments and timestamps.
    
    Args:
        request (PredictWithTimestampsRequest): List of comments with timestamps
        
    Returns:
        List[PredictWithTimestampsResponse]: List of predictions with comments and timestamps
        
    Raises:
        HTTPException: If no comments provided or prediction fails
        
    Example:
        Request:
        ```json
        {
            "comments": [
                {"text": "Great video!", "timestamp": "2024-01-15T10:30:00"},
                {"text": "Not good", "timestamp": "2024-01-15T11:00:00"}
            ]
        }
        ```
        Response:
        ```json
        [
            {"comment": "Great video!", "sentiment": "1", "timestamp": "2024-01-15T10:30:00"},
            {"comment": "Not good", "sentiment": "-1", "timestamp": "2024-01-15T11:00:00"}
        ]
        ```
    """
    # Check if models are loaded
    if model is None or vectorizer is None:
        logger.error("Models not loaded - cannot make predictions")
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please check server logs and ensure model files exist."
        )
    
    # Extract comments and timestamps from request
    comments_data = request.comments
    
    if not comments_data:
        raise HTTPException(status_code=400, detail="No comments provided")

    try:
        # Extract text and timestamps from comment items
        comments = [item.text for item in comments_data]
        timestamps = [item.timestamp for item in comments_data]

        # Preprocess each comment before vectorizing
        preprocessed_comments = [preprocess_comment(comment) for comment in comments]
        
        # Transform comments using the TF-IDF vectorizer
        transformed_comments = vectorizer.transform(preprocessed_comments)

        # Convert the sparse matrix to dense format for model prediction
        dense_comments = transformed_comments.toarray()
        
        # Make predictions using the LGBM model
        predictions = model.predict(dense_comments).tolist()
        
        # Convert predictions to strings for consistency in response
        predictions = [str(pred) for pred in predictions]
        
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
    
    # Build response with original comments, predicted sentiments, and timestamps
    response = [
        {
            "comment": comment,
            "sentiment": sentiment,
            "timestamp": timestamp
        }
        for comment, sentiment, timestamp in zip(comments, predictions, timestamps)
    ]
    
    return response


@app.post("/predict", response_model=List[PredictResponse], tags=["Prediction"])
async def predict(request: PredictRequest):
    """
    Predict sentiment for a list of comments.
    
    This endpoint analyzes a list of comments and returns sentiment predictions
    along with the original comments.
    
    Args:
        request (PredictRequest): List of comment texts
        
    Returns:
        List[PredictResponse]: List of predictions with original comments
        
    Raises:
        HTTPException: If no comments provided or prediction fails
        
    Example:
        Request:
        ```json
        {
            "comments": ["Great video!", "Not good", "Amazing content"]
        }
        ```
        Response:
        ```json
        [
            {"comment": "Great video!", "sentiment": "1"},
            {"comment": "Not good", "sentiment": "-1"},
            {"comment": "Amazing content", "sentiment": "1"}
        ]
        ```
    """
    # Check if models are loaded
    if model is None or vectorizer is None:
        logger.error("Models not loaded - cannot make predictions")
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please check server logs and ensure model files exist."
        )
    
    # Extract comments from request
    comments = request.comments
    logger.info(f"Received {len(comments)} comments for prediction")
    logger.debug(f"Comment type: {type(comments)}")
    
    if not comments:
        raise HTTPException(status_code=400, detail="No comments provided")

    try:
        # Preprocess each comment before vectorizing
        preprocessed_comments = [preprocess_comment(comment) for comment in comments]
        
        # Transform comments using the TF-IDF vectorizer
        transformed_comments = vectorizer.transform(preprocessed_comments)

        # Convert the sparse matrix to dense format for model prediction
        dense_comments = transformed_comments.toarray()
        
        # Make predictions using the LGBM model
        predictions = model.predict(dense_comments).tolist()
        
        # Note: Predictions are already integers, converting to string if needed
        # predictions = [str(pred) for pred in predictions]
        
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
    
    # Build response with original comments and predicted sentiments
    response = [
        {
            "comment": comment,
            "sentiment": str(sentiment)  # Convert to string for consistency
        }
        for comment, sentiment in zip(comments, predictions)
    ]
    
    return response


@app.post("/generate_chart", tags=["Visualization"])
async def generate_chart(request: SentimentCountsRequest):
    """
    Generate a pie chart showing sentiment distribution.
    
    Creates a visual pie chart representing the distribution of positive,
    neutral, and negative sentiments from the provided counts.
    
    Args:
        request (SentimentCountsRequest): Dictionary with sentiment counts
            Expected keys: '1' (positive), '0' (neutral), '-1' (negative)
        
    Returns:
        StreamingResponse: PNG image of the pie chart
        
    Raises:
        HTTPException: If no sentiment counts provided or generation fails
        
    Example:
        Request:
        ```json
        {
            "sentiment_counts": {
                "1": 150,
                "0": 50,
                "-1": 30
            }
        }
        ```
    """
    try:
        sentiment_counts = request.sentiment_counts
        
        if not sentiment_counts:
            raise HTTPException(status_code=400, detail="No sentiment counts provided")

        # Prepare data for the pie chart
        labels = ['Positive', 'Neutral', 'Negative']
        sizes = [
            int(sentiment_counts.get('1', 0)),   # Positive sentiment count
            int(sentiment_counts.get('0', 0)),   # Neutral sentiment count
            int(sentiment_counts.get('-1', 0))   # Negative sentiment count
        ]
        
        # Validate that we have data to plot
        if sum(sizes) == 0:
            raise ValueError("Sentiment counts sum to zero")
        
        # Define colors: Blue for Positive, Gray for Neutral, Red for Negative
        colors = ['#36A2EB', '#C9CBCF', '#FF6384']

        # Generate the pie chart
        plt.figure(figsize=(6, 6))
        plt.pie(
            sizes,
            labels=labels,
            colors=colors,
            autopct='%1.1f%%',  # Show percentage with 1 decimal place
            startangle=140,      # Start angle for first slice
            textprops={'color': 'w'}  # White text color
        )
        plt.axis('equal')  # Equal aspect ratio ensures pie is drawn as a circle

        # Save the chart to a BytesIO object (in-memory file)
        img_io = io.BytesIO()
        plt.savefig(img_io, format='PNG', transparent=True)
        img_io.seek(0)  # Reset pointer to beginning of file
        plt.close()  # Close the plot to free memory

        # Return the image as a streaming response
        return StreamingResponse(img_io, media_type="image/png")
        
    except Exception as e:
        logger.error(f"Error in /generate_chart: {e}")
        raise HTTPException(status_code=500, detail=f"Chart generation failed: {str(e)}")


@app.post("/generate_wordcloud", tags=["Visualization"])
async def generate_wordcloud(request: WordCloudRequest):
    """
    Generate a word cloud from comments.
    
    Creates a visual word cloud showing the most frequent words in the
    provided comments after preprocessing.
    
    Args:
        request (WordCloudRequest): List of comment texts
        
    Returns:
        StreamingResponse: PNG image of the word cloud
        
    Raises:
        HTTPException: If no comments provided or generation fails
        
    Example:
        Request:
        ```json
        {
            "comments": ["Great video!", "Love this content", "Amazing work"]
        }
        ```
    """
    try:
        comments = request.comments

        if not comments:
            raise HTTPException(status_code=400, detail="No comments provided")

        # Preprocess all comments to clean and normalize text
        preprocessed_comments = [preprocess_comment(comment) for comment in comments]

        # Combine all preprocessed comments into a single string
        text = ' '.join(preprocessed_comments)

        # Generate the word cloud with custom styling
        wordcloud = WordCloud(
            width=800,                              # Width of the image
            height=400,                             # Height of the image
            background_color='black',               # Black background
            colormap='Blues',                       # Blue color scheme
            stopwords=set(stopwords.words('english')),  # Remove common words
            collocations=False                      # Don't include bigrams
        ).generate(text)

        # Save the word cloud to a BytesIO object (in-memory file)
        img_io = io.BytesIO()
        wordcloud.to_image().save(img_io, format='PNG')
        img_io.seek(0)  # Reset pointer to beginning of file

        # Return the image as a streaming response
        return StreamingResponse(img_io, media_type="image/png")
        
    except Exception as e:
        logger.error(f"Error in /generate_wordcloud: {e}")
        raise HTTPException(status_code=500, detail=f"Word cloud generation failed: {str(e)}")


@app.post("/generate_trend_graph", tags=["Visualization"])
async def generate_trend_graph(request: TrendGraphRequest):
    """
    Generate a trend graph showing sentiment changes over time.
    
    Creates a line graph showing the monthly percentage of positive, neutral,
    and negative sentiments over time.
    
    Args:
        request (TrendGraphRequest): List of sentiment data with timestamps
        
    Returns:
        StreamingResponse: PNG image of the trend graph
        
    Raises:
        HTTPException: If no sentiment data provided or generation fails
        
    Example:
        Request:
        ```json
        {
            "sentiment_data": [
                {"sentiment": 1, "timestamp": "2024-01-15T10:30:00"},
                {"sentiment": -1, "timestamp": "2024-01-16T11:00:00"},
                {"sentiment": 0, "timestamp": "2024-02-10T14:20:00"}
            ]
        }
        ```
    """
    try:
        sentiment_data = request.sentiment_data

        if not sentiment_data:
            raise HTTPException(status_code=400, detail="No sentiment data provided")

        # Convert sentiment data to DataFrame for easier manipulation
        df = pd.DataFrame([item.dict() for item in sentiment_data])
        
        # Parse timestamps to datetime objects
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Set the timestamp as the index for time-based operations
        df.set_index('timestamp', inplace=True)

        # Ensure the 'sentiment' column is numeric (should be -1, 0, or 1)
        df['sentiment'] = df['sentiment'].astype(int)

        # Map sentiment values to human-readable labels
        sentiment_labels = {-1: 'Negative', 0: 'Neutral', 1: 'Positive'}

        # Resample the data by month and count occurrences of each sentiment
        monthly_counts = df.resample('M')['sentiment'].value_counts().unstack(fill_value=0)

        # Calculate total counts per month for percentage calculation
        monthly_totals = monthly_counts.sum(axis=1)

        # Calculate percentages for each sentiment per month
        monthly_percentages = (monthly_counts.T / monthly_totals).T * 100

        # Ensure all sentiment columns are present (even if some months have no data)
        for sentiment_value in [-1, 0, 1]:
            if sentiment_value not in monthly_percentages.columns:
                monthly_percentages[sentiment_value] = 0

        # Sort columns by sentiment value for consistent ordering
        monthly_percentages = monthly_percentages[[-1, 0, 1]]

        # Create the line plot
        plt.figure(figsize=(12, 6))

        # Define colors for each sentiment
        colors = {
            -1: 'red',     # Negative sentiment
            0: 'gray',     # Neutral sentiment
            1: 'green'     # Positive sentiment
        }

        # Plot a line for each sentiment type
        for sentiment_value in [-1, 0, 1]:
            plt.plot(
                monthly_percentages.index,
                monthly_percentages[sentiment_value],
                marker='o',                              # Circular markers at data points
                linestyle='-',                           # Solid line
                label=sentiment_labels[sentiment_value], # Legend label
                color=colors[sentiment_value]            # Line color
            )

        # Customize the plot
        plt.title('Monthly Sentiment Percentage Over Time')
        plt.xlabel('Month')
        plt.ylabel('Percentage of Comments (%)')
        plt.grid(True)                                   # Add grid for readability
        plt.xticks(rotation=45)                          # Rotate x-axis labels

        # Format the x-axis dates to show year-month
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=12))

        plt.legend()           # Show legend
        plt.tight_layout()     # Adjust layout to prevent label cutoff

        # Save the trend graph to a BytesIO object (in-memory file)
        img_io = io.BytesIO()
        plt.savefig(img_io, format='PNG')
        img_io.seek(0)  # Reset pointer to beginning of file
        plt.close()     # Close the plot to free memory

        # Return the image as a streaming response
        return StreamingResponse(img_io, media_type="image/png")
        
    except Exception as e:
        logger.error(f"Error in /generate_trend_graph: {e}")
        raise HTTPException(status_code=500, detail=f"Trend graph generation failed: {str(e)}")


# ========== Application Entry Point ==========

if __name__ == '__main__':
    import uvicorn
    # Run the FastAPI application using uvicorn server
    # host='0.0.0.0' makes it accessible from any network interface
    # port=5000 to match the original Flask port
    # reload=True enables auto-reload during development
    uvicorn.run("fastAPI_app.main:app", host='0.0.0.0', port=5000, reload=True)