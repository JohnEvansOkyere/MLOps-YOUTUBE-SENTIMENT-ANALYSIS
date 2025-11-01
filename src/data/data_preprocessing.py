# src/data/data_preprocessing.py
"""
Data Preprocessing Pipeline
Cleans, normalizes, and validates text data for sentiment analysis
"""

import numpy as np
import pandas as pd
import os
import re
import nltk
import string
import logging
from pathlib import Path
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# ==================== LOGGING SETUP ====================
def setup_logging() -> logging.Logger:
    """Configure logging."""
    logger = logging.getLogger("data_preprocessing")
    logger.setLevel(logging.DEBUG)

    logger.handlers = []

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    file_handler = logging.FileHandler(log_dir / "preprocessing_errors.log")
    file_handler.setLevel(logging.ERROR)

    detailed_handler = logging.FileHandler(log_dir / "preprocessing.log")
    detailed_handler.setLevel(logging.DEBUG)

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

# Download required NLTK data
try:
    nltk.data.find("corpora/wordnet")
    nltk.data.find("corpora/stopwords")
except LookupError:
    logger.info("Downloading NLTK data...")
    nltk.download("wordnet", quiet=True)
    nltk.download("stopwords", quiet=True)


# ==================== PREPROCESSING FUNCTIONS ====================
def preprocess_comment(comment):
    """Apply preprocessing transformations to a comment."""
    try:
        # Handle non-string inputs
        if not isinstance(comment, str):
            return ""

        # Convert to lowercase
        comment = comment.lower()

        # Remove trailing and leading whitespaces
        comment = comment.strip()

        # Remove newline characters
        comment = re.sub(r"\n", " ", comment)

        # Remove non-alphanumeric characters, except punctuation
        comment = re.sub(r"[^A-Za-z0-9\s!?.,]", "", comment)

        # Remove stopwords but retain important ones for sentiment analysis
        stop_words = set(stopwords.words("english")) - {
            "not",
            "but",
            "however",
            "no",
            "yet",
        }
        comment = " ".join([word for word in comment.split() if word not in stop_words])

        # Lemmatize the words
        lemmatizer = WordNetLemmatizer()
        comment = " ".join([lemmatizer.lemmatize(word) for word in comment.split()])

        return comment

    except Exception as e:
        logger.error(f"Error preprocessing comment: {e}")
        return ""


def validate_data(df: pd.DataFrame, dataset_name: str = "dataset") -> pd.DataFrame:
    """
    Validate and log data statistics.

    Args:
        df: DataFrame to validate
        dataset_name: Name for logging

    Returns:
        Validated DataFrame
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"VALIDATING {dataset_name.upper()}")
    logger.info(f"{'='*60}")

    # Initial stats
    logger.info(f"Initial shape: {df.shape}")
    logger.info(f"Initial rows: {len(df)}")

    # Check for required columns
    required_cols = ["comment", "category"]  # Adjust based on your data
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        # Try alternative column names
        if "comment" not in df.columns and "text" in df.columns:
            df["comment"] = df["text"]
            logger.info("Renamed 'text' column to 'comment'")
        elif "comment" not in df.columns and "clean_comment" in df.columns:
            df["comment"] = df["clean_comment"]
            logger.info("Using existing 'clean_comment' as 'comment'")

    logger.info(f"Columns: {list(df.columns)}")

    return df


def clean_data(df: pd.DataFrame, dataset_name: str = "dataset") -> pd.DataFrame:
    """
    Clean data by handling missing values and duplicates.

    Args:
        df: DataFrame to clean
        dataset_name: Name for logging

    Returns:
        Cleaned DataFrame
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"CLEANING {dataset_name.upper()}")
    logger.info(f"{'='*60}")

    initial_rows = len(df)

    # 1. Handle missing values
    missing_before = df.isna().sum().sum()
    logger.info(f"Missing values before: {missing_before}")

    if missing_before > 0:
        # Log which columns have missing values
        missing_by_col = df.isna().sum()
        for col, count in missing_by_col[missing_by_col > 0].items():
            logger.info(f"  - {col}: {count} missing")

        # Fill missing comments with empty string
        if "comment" in df.columns:
            df["comment"].fillna("", inplace=True)
        if "clean_comment" in df.columns:
            df["clean_comment"].fillna("", inplace=True)

        # Drop rows with missing target variable
        if "category" in df.columns:
            df = df.dropna(subset=["category"])

        missing_after = df.isna().sum().sum()
        logger.info(f"Missing values after: {missing_after}")
        logger.info(f"Rows after handling missing: {len(df)}")

    # 2. Remove duplicates
    duplicates_before = df.duplicated().sum()
    logger.info(f"\nDuplicates before: {duplicates_before}")

    if duplicates_before > 0:
        # Check duplicates on comment column if it exists
        if "comment" in df.columns:
            df = df.drop_duplicates(subset=["comment"], keep="first")
        else:
            df = df.drop_duplicates()

        duplicates_after = df.duplicated().sum()
        logger.info(f"Duplicates after: {duplicates_after}")
        logger.info(f"Rows removed: {duplicates_before}")

    # 3. Remove empty comments
    if "comment" in df.columns:
        empty_before = (df["comment"].str.strip() == "").sum()
        logger.info(f"\nEmpty comments before: {empty_before}")

        if empty_before > 0:
            df = df[df["comment"].str.strip() != ""]
            logger.info(f"Empty comments removed: {empty_before}")

    # 4. Reset index
    df = df.reset_index(drop=True)

    final_rows = len(df)
    rows_removed = initial_rows - final_rows

    logger.info(f"\nCleaning Summary:")
    logger.info(f"  Initial rows: {initial_rows}")
    logger.info(f"  Final rows: {final_rows}")
    logger.info(
        f"  Total removed: {rows_removed} ({100*rows_removed/initial_rows:.2f}%)"
    )

    return df


def normalize_text(df: pd.DataFrame, dataset_name: str = "dataset") -> pd.DataFrame:
    """
    Apply preprocessing to the text data in the dataframe.

    Args:
        df: DataFrame with 'comment' column
        dataset_name: Name for logging

    Returns:
        DataFrame with 'clean_comment' column
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"NORMALIZING TEXT - {dataset_name.upper()}")
    logger.info(f"{'='*60}")

    try:
        # Ensure comment column exists
        if "comment" not in df.columns:
            raise ValueError("DataFrame must have 'comment' column")

        # Apply preprocessing
        logger.info(f"Processing {len(df)} comments...")
        df["clean_comment"] = df["comment"].apply(preprocess_comment)

        # Remove comments that became empty after preprocessing
        empty_after = (df["clean_comment"].str.strip() == "").sum()
        if empty_after > 0:
            logger.warning(
                f"Found {empty_after} comments that became empty after preprocessing"
            )
            df = df[df["clean_comment"].str.strip() != ""]
            df = df.reset_index(drop=True)

        # Log statistics
        avg_length_before = df["comment"].str.len().mean()
        avg_length_after = df["clean_comment"].str.len().mean()

        logger.info(f"✓ Text normalization completed")
        logger.info(f"  Rows processed: {len(df)}")
        logger.info(f"  Avg length before: {avg_length_before:.1f} chars")
        logger.info(f"  Avg length after: {avg_length_after:.1f} chars")
        logger.info(
            f"  Reduction: {100*(avg_length_before-avg_length_after)/avg_length_before:.1f}%"
        )

        return df

    except Exception as e:
        logger.error(f"Error during text normalization: {e}")
        raise


def save_data(
    train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str = "./data"
) -> None:
    """Save the processed train and test datasets."""
    try:
        interim_data_path = Path(data_path) / "interim"
        interim_data_path.mkdir(parents=True, exist_ok=True)

        train_path = interim_data_path / "train_processed.csv"
        test_path = interim_data_path / "test_processed.csv"

        train_data.to_csv(train_path, index=False)
        test_data.to_csv(test_path, index=False)

        logger.info(f"\n✓ Processed data saved:")
        logger.info(f"  Train: {train_path} ({len(train_data)} rows)")
        logger.info(f"  Test: {test_path} ({len(test_data)} rows)")

    except Exception as e:
        logger.error(f"Error saving data: {e}")
        raise


# ==================== MAIN ====================
def main():
    """Main preprocessing pipeline."""
    try:
        logger.info(f"\n{'='*60}")
        logger.info("DATA PREPROCESSING PIPELINE")
        logger.info(f"{'='*60}\n")

        # Load raw data
        logger.info("Loading raw data...")
        train_data = pd.read_csv("./data/raw/train.csv")
        test_data = pd.read_csv("./data/raw/test.csv")
        logger.info(f"✓ Data loaded")
        logger.info(f"  Train: {train_data.shape}")
        logger.info(f"  Test: {test_data.shape}")

        # Validate data
        train_data = validate_data(train_data, "train")
        test_data = validate_data(test_data, "test")

        # Clean data (remove duplicates, handle missing values)
        train_data = clean_data(train_data, "train")
        test_data = clean_data(test_data, "test")

        # Normalize text
        train_data = normalize_text(train_data, "train")
        test_data = normalize_text(test_data, "test")

        # Final validation
        logger.info(f"\n{'='*60}")
        logger.info("FINAL VALIDATION")
        logger.info(f"{'='*60}")
        logger.info(f"Train shape: {train_data.shape}")
        logger.info(f"Test shape: {test_data.shape}")
        logger.info(f"Train missing values: {train_data.isna().sum().sum()}")
        logger.info(f"Test missing values: {test_data.isna().sum().sum()}")
        logger.info(f"Train duplicates: {train_data.duplicated().sum()}")
        logger.info(f"Test duplicates: {test_data.duplicated().sum()}")

        # Save processed data
        save_data(train_data, test_data, data_path="./data")

        logger.info(f"\n{'='*60}")
        logger.info("✅ PREPROCESSING COMPLETED SUCCESSFULLY")
        logger.info(f"{'='*60}\n")

    except Exception as e:
        logger.error(f"\n❌ Preprocessing failed: {e}", exc_info=True)
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)
