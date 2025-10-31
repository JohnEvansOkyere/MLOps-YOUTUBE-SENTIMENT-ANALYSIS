# scripts/create_reference_data.py
"""
Create reference dataset for drift detection
"""

import pandas as pd
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def create_reference_data(
    train_data_path: str = "data/interim/train_processed.csv",
    output_path: str = "reference_data/train_reference.csv",
    sample_size: int = 5000
):
    """Create reference dataset from training data."""
    
    logger.info(f"Creating reference data from {train_data_path}")
    
    # Load training data
    df = pd.read_csv(train_data_path)
    
    # Sample if needed
    if len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42)
    
    # Create output directory
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save reference data
    df.to_csv(output_path, index=False)
    
    logger.info(f"✓ Reference data saved to {output_path}")
    logger.info(f"  Shape: {df.shape}")
    
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    create_reference_data()