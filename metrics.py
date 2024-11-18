import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr
from typing import Tuple

def calculate_metrics(labels: np.ndarray, outputs: np.ndarray) -> Tuple[float, float, float]:
    """
    Calculates the evaluation metrics for model performance.

    Args:
        labels (np.ndarray): True labels/ground truth values.
        outputs (np.ndarray): Model predicted outputs.

    Returns:
        Tuple[float, float, float]: A tuple containing three metrics:
            - Mean Absolute Error (MAE)
            - Mean Squared Error (MSE)
            - Pearson Correlation Coefficient
    """
    # Calculate Mean Absolute Error
    mae = mean_absolute_error(labels, outputs)

    # Calculate Mean Squared Error
    mse = mean_squared_error(labels, outputs)

    # Calculate Pearson Correlation Coefficient
    pearson_corr, _ = pearsonr(labels, outputs)

    return mae, mse, pearson_corr
