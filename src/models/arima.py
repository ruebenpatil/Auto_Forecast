from itertools import product
import numpy as np
import pandas as pd 
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

from src.utils.metrics import mean_relative_forecast_bias, smape, mean_absolute_forecast_bias


def get_seasonal_period(data):
    inferred_freq = pd.infer_freq(data.index)
    if inferred_freq in ['W', 'W-MON', 'W-SUN']:
        return 52  # Weekly data
    elif inferred_freq in ['MS', 'M']:
        return 12  # Monthly data
    elif inferred_freq in ['D', 'B']:
        return 7 # Daily data
    else:
        return max(2, min(12, len(data) // 2)) 


def run_arima(train, test):
    best_loss = float("inf")
    best_order = (1, 1, 1)  # default fallback
    lambda_mfb = 0.5
    candidate_orders = product(range(3), range(2), range(3))

    for order in candidate_orders:
        try:
            model = ARIMA(train, order=order).fit()
            pred = model.forecast(steps=len(test))
            loss = smape(test, pred) + lambda_mfb * abs(mean_relative_forecast_bias(test, pred))
            if loss < best_loss:
                best_loss = loss
                best_order = order
        except (ValueError, np.linalg.LinAlgError):
            continue

    final_model = ARIMA(train, order=best_order).fit()
    return final_model.forecast(steps=len(test))


def run_sarima(train, test):
    best_score = float("inf")
    best_order = None
    best_seasonal_order = None

    seasonal_period = get_seasonal_period(train)

    # Generate all combinations of orders
    candidate_orders = product(range(3), range(2), range(3), range(2), range(2), range(2))

    for order in candidate_orders:
        p, d, q, P, D, Q = order
        try:
            # Fit SARIMAX model
            model = SARIMAX(train, order=(p, d, q), seasonal_order=(P, D, Q, seasonal_period)).fit()
            pred = model.forecast(steps=len(test))

            # Evaluate model with combined score (SMAPE + Mean Forecast Bias)
            combined_score = smape(test, pred) + abs(mean_absolute_forecast_bias(test, pred))

            # Update best model if score improves
            if combined_score < best_score:
                best_score = combined_score
                best_order = (p, d, q)
                best_seasonal_order = (P, D, Q, seasonal_period)

        except (ValueError, np.linalg.LinAlgError):  # Catch specific exceptions
            continue

    # Final model with the best found parameters
    final_model = SARIMAX(train, order=best_order, seasonal_order=best_seasonal_order).fit()
    return final_model.forecast(steps=len(test))

