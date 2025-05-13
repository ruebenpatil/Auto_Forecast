import optuna
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from src.utils.metrics import smape

def run_holt_winters(train, test, n_trials=10):
    train_series = pd.Series(train)

    # Function to fit and forecast using Holt-Winters model
    def fit_holt_winters_model(train_series, trend, seasonal, seasonal_periods, smoothing_level, smoothing_slope, smoothing_seasonal):
        try:
            model = ExponentialSmoothing(
                train_series,
                trend=trend,
                seasonal=seasonal,
                seasonal_periods=seasonal_periods,
            )
            fit_model = model.fit(
                smoothing_level=smoothing_level,
                smoothing_slope=smoothing_slope,
                smoothing_seasonal=smoothing_seasonal,
            )
            return fit_model.forecast(len(test))
        except (ValueError, TypeError):  # Handle specific exceptions
            return None  # Return None if model fitting fails

    # Objective function for Optuna optimization
    def objective(trial):
        trend = trial.suggest_categorical("trend", ["add", "mul", None])
        seasonal = trial.suggest_categorical("seasonal", ["add", "mul", None])
        seasonal_periods = trial.suggest_int("seasonal_periods", 2, min(30, len(train) // 2))
        smoothing_level = trial.suggest_float("smoothing_level", 0.01, 1.0)
        smoothing_slope = trial.suggest_float("smoothing_slope", 0.01, 1.0)
        smoothing_seasonal = trial.suggest_float("smoothing_seasonal", 0.01, 1.0)

        # Fit the model and get forecast
        forecast = fit_holt_winters_model(
            train_series,
            trend,
            seasonal,
            seasonal_periods,
            smoothing_level,
            smoothing_slope,
            smoothing_seasonal
        )

        # Return SMAPE if forecast is successful, otherwise return a large value
        if forecast is None:
            return float("inf")
        return smape(test, forecast)

    # Perform hyperparameter optimization
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, timeout=60, show_progress_bar=True)

    # Get best parameters and fit the final model
    best_params = study.best_params
    best_forecast = fit_holt_winters_model(
        train_series,
        trend=best_params["trend"],
        seasonal=best_params["seasonal"],
        seasonal_periods=best_params["seasonal_periods"],
        smoothing_level=best_params["smoothing_level"],
        smoothing_slope=best_params["smoothing_slope"],
        smoothing_seasonal=best_params["smoothing_seasonal"]
    )

    return best_forecast
