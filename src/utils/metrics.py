import numpy as np 

def smape(actual: np.ndarray, predicted: np.ndarray) -> float:
     """Symmetric Mean Absolute Percentage Error"""
     return float(np.mean(200 * np.abs(actual - predicted) / (np.abs(actual) + np.abs(predicted))))


def evaluate_model(actual, predicted):
        """Calculate SMAPE and Mean Forecast Bias"""
        return smape(actual, predicted), mean_absolute_forecast_bias(actual, predicted)

def mean_absolute_forecast_bias(actual, predicted) -> float:
    """Mean Absolute Forecast Bias (raw difference)"""
    return float(np.mean(predicted - actual))

def mean_relative_forecast_bias(actual, predicted) -> float:
    """Mean Relative Forecast Bias (percentage-based)"""
    return float(100 * np.mean((predicted - actual) / actual))
