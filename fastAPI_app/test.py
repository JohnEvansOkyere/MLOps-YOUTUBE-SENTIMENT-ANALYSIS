"""
Test script for YouTube Sentiment Analysis Model.

This script tests the sentiment analysis model locally before deploying to the API.
It validates that the model, vectorizer, and preprocessing pipeline work correctly.
"""

import mlflow
import numpy as np
import joblib
import re
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from mlflow.tracking import MlflowClient
import pickle
import logging
from typing import List, Dict, Tuple, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ========== Preprocessing Functions ==========

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


# ========== Model Loading Functions ==========

def load_model_and_vectorizer(
    model_name: str,
    model_version: str,
    vectorizer_path: str
) -> Tuple[Any, Any]:
    """
    Load model from MLflow model registry and vectorizer from local storage.
    
    This function connects to a remote MLflow tracking server and loads
    a registered model along with its TF-IDF vectorizer.
    
    Args:
        model_name (str): Name of the model in MLflow registry
        model_version (str): Version of the model to load (e.g., "1", "2")
        vectorizer_path (str): Path to the pickled TF-IDF vectorizer file
        
    Returns:
        tuple: (model, vectorizer) - Loaded model and vectorizer objects
        
    Note:
        Requires network access to the MLflow tracking server
    """
    # Set MLflow tracking URI to your remote server
    mlflow.set_tracking_uri("http://ec2-54-167-108-249.compute-1.amazonaws.com:5000/")
    logger.info(f"Connecting to MLflow server...")
    
    # Initialize MLflow client
    client = MlflowClient()
    
    # Construct model URI from registry
    model_uri = f"models:/{model_name}/{model_version}"
    logger.info(f"Loading model from: {model_uri}")
    
    # Load the model from MLflow registry
    model = mlflow.pyfunc.load_model(model_uri)
    
    # Load the TF-IDF vectorizer from local pickle file
    logger.info(f"Loading vectorizer from: {vectorizer_path}")
    with open(vectorizer_path, 'rb') as file:
        vectorizer = pickle.load(file)
    
    logger.info("Model and vectorizer loaded successfully from MLflow")
    return model, vectorizer


def load_model(model_path: str, vectorizer_path: str) -> Tuple[Any, Any]:
    """
    Load the trained model and vectorizer from local pickle files.
    
    This is the local loading method that doesn't require MLflow server access.
    Use this for testing and development when MLflow is not available.
    
    Args:
        model_path (str): Path to the pickled model file (e.g., "./lgbm_model.pkl")
        vectorizer_path (str): Path to the pickled TF-IDF vectorizer file
        
    Returns:
        tuple: (model, vectorizer) - Loaded model and vectorizer objects
        
    Raises:
        Exception: If loading fails (file not found, corrupted pickle, etc.)
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
        
        logger.info("Model and vectorizer loaded successfully from local files")
        return model, vectorizer
        
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading model or vectorizer: {e}")
        raise


# ========== Model Testing Functions ==========

def test_predict(
    model: Any,
    vectorizer: Any,
    test_comments: List[str] = None
) -> List[Dict[str, Any]]:
    """
    Test the sentiment prediction pipeline with sample comments.
    
    This function runs a complete prediction workflow:
    1. Validates input comments
    2. Preprocesses each comment
    3. Vectorizes using TF-IDF
    4. Makes predictions with the model
    5. Returns results with sentiment labels
    
    Args:
        model: Trained sentiment analysis model
        vectorizer: Fitted TF-IDF vectorizer
        test_comments (List[str], optional): Comments to test.
            Defaults to sample positive and negative comments.
            
    Returns:
        List[Dict[str, Any]]: List of dictionaries containing comments and predictions
        
    Example:
        >>> results = test_predict(model, vectorizer, ["Great video!", "Terrible"])
        >>> print(results)
        [{'comment': 'Great video!', 'sentiment': 1}, {'comment': 'Terrible', 'sentiment': -1}]
    """
    # Use default test comments if none provided
    if test_comments is None:
        test_comments = [
            "I love this product!",
            "This is the worst experience.",
            "It's okay, nothing special.",
            "Amazing quality and fast delivery!",
            "Terrible customer service, never buying again."
        ]
    
    # Validate that we have comments to process
    if not test_comments:
        logger.error("No comments provided for testing")
        return {"error": "No comments provided"}
    
    logger.info(f"Testing prediction with {len(test_comments)} comments")
    
    try:
        # Step 1: Preprocess each comment before vectorizing
        logger.info("Preprocessing comments...")
        preprocessed_comments = [preprocess_comment(comment) for comment in test_comments]
        logger.debug(f"Preprocessed comments: {preprocessed_comments}")
        
        # Step 2: Transform comments using the TF-IDF vectorizer
        logger.info("Vectorizing comments...")
        transformed_comments = vectorizer.transform(preprocessed_comments)
        
        # Step 3: Convert the sparse matrix to dense format for model input
        dense_comments = transformed_comments.toarray()
        logger.debug(f"Dense matrix shape: {dense_comments.shape}")
        
        # Step 4: Make predictions using the trained model
        logger.info("Making predictions...")
        predictions = model.predict(dense_comments).tolist()
        
        # Optional: Convert predictions to strings for consistency with API
        # predictions = [str(pred) for pred in predictions]
        
        logger.info("Predictions completed successfully")
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise
    
    # Build response with original comments and predicted sentiments
    # Sentiment mapping: -1 = Negative, 0 = Neutral, 1 = Positive
    response = [
        {
            "comment": comment,
            "sentiment": sentiment,
            "sentiment_label": get_sentiment_label(sentiment)
        }
        for comment, sentiment in zip(test_comments, predictions)
    ]
    
    return response


def get_sentiment_label(sentiment: int) -> str:
    """
    Convert numeric sentiment to human-readable label.
    
    Args:
        sentiment (int): Numeric sentiment value (-1, 0, or 1)
        
    Returns:
        str: Human-readable sentiment label
    """
    sentiment_map = {
        -1: "Negative",
        0: "Neutral",
        1: "Positive"
    }
    return sentiment_map.get(sentiment, "Unknown")


def print_results(results: List[Dict[str, Any]]) -> None:
    """
    Pretty print prediction results to console.
    
    Args:
        results (List[Dict]): List of prediction results
    """
    print("\n" + "="*80)
    print("SENTIMENT ANALYSIS RESULTS")
    print("="*80)
    
    for i, result in enumerate(results, 1):
        sentiment_emoji = {
            "Positive": "😊",
            "Neutral": "😐",
            "Negative": "😞"
        }
        emoji = sentiment_emoji.get(result['sentiment_label'], "")
        
        print(f"\n{i}. Comment: {result['comment']}")
        print(f"   Sentiment: {result['sentiment_label']} ({result['sentiment']}) {emoji}")
    
    print("\n" + "="*80)
    
    # Print summary statistics
    sentiment_counts = {}
    for result in results:
        label = result['sentiment_label']
        sentiment_counts[label] = sentiment_counts.get(label, 0) + 1
    
    print("\nSummary:")
    for label, count in sentiment_counts.items():
        percentage = (count / len(results)) * 100
        print(f"  {label}: {count} ({percentage:.1f}%)")
    print("="*80 + "\n")


# ========== Main Execution ==========

def main():
    """
    Main function to run the sentiment analysis test.
    
    This function:
    1. Loads the model and vectorizer
    2. Tests predictions with sample comments
    3. Displays results in a formatted output
    """
    try:
        logger.info("Starting sentiment analysis test...")
        
        # Load model and vectorizer from local pickle files
        # For MLflow loading, uncomment the line below and comment out the local loading
        model, vectorizer = load_model("./lgbm_model.pkl", "./tfidf_vectorizer.pkl")
        
        # Alternative: Load from MLflow model registry
        # model, vectorizer = load_model_and_vectorizer("my_model", "1", "./tfidf_vectorizer.pkl")
        
        # Define test comments (you can modify these)
        test_comments = [
            "I love this product!",
            "This is the worst experience.",
            "It's okay, nothing special.",
            "Amazing quality and fast delivery!",
            "Terrible customer service, never buying again.",
            "Not bad, but could be better.",
            "Absolutely fantastic! Highly recommend!",
            "Disappointed with the quality."
        ]
        
        # Run prediction test
        results = test_predict(model, vectorizer, test_comments)
        
        # Display results
        print_results(results)
        
        logger.info("Test completed successfully!")
        
    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        raise


if __name__ == '__main__':
    # Execute the main test function
    main()