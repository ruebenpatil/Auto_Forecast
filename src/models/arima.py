from itertools import product
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import optuna

from src.utils.metrics import (
    mean_relative_forecast_bias,
    smape,
    mean_absolute_forecast_bias,
)


def get_seasonal_period(data):
    inferred_freq = pd.infer_freq(data.index)
    if inferred_freq in ["W", "W-MON", "W-SUN"]:
        return 52  # Weekly data
    elif inferred_freq in ["MS", "M"]:
        return 12  # Monthly data
    elif inferred_freq in ["D", "B"]:
        return 7  # Daily data
    else:
        return max(2, min(12, len(data) // 2))


def run_arima(train, test, n_trials=60, lambda_mfb=0.5):
    def objective(trial):
        p = trial.suggest_int("p", 0, 3)
        d = trial.suggest_int("d", 0, 2)
        q = trial.suggest_int("q", 0, 3)

        try:
            model = ARIMA(train, order=(p, d, q)).fit()
            pred = model.forecast(steps=len(test))
            loss = smape(test, pred) + lambda_mfb * abs(
                mean_relative_forecast_bias(test, pred)
            )
            return loss
        except (ValueError, np.linalg.LinAlgError):
            return float("inf")  # Penalize failed trials

    # Run Optuna optimization
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)

    # Fit final model using best parameters
    best_params = study.best_params
    final_model = ARIMA(
        train, order=(best_params["p"], best_params["d"], best_params["q"])
    ).fit()
    return final_model.forecast(steps=len(test))


def run_sarima(train, test, n_trials=60):
    seasonal_period = get_seasonal_period(train)  # Your existing function

    def objective(trial):
        p = trial.suggest_int("p", 0, 3)
        d = trial.suggest_int("d", 0, 2)
        q = trial.suggest_int("q", 0, 3)
        P = trial.suggest_int("P", 0, 2)
        D = trial.suggest_int("D", 0, 2)
        Q = trial.suggest_int("Q", 0, 2)

        try:
            model = SARIMAX(
                train,
                order=(p, d, q),
                seasonal_order=(P, D, Q, seasonal_period),
                enforce_stationarity=False,
                enforce_invertibility=False,
            ).fit(disp=False)

            pred = model.forecast(steps=len(test))
            score = smape(test, pred) + abs(mean_absolute_forecast_bias(test, pred))
            return score

        except (ValueError, np.linalg.LinAlgError):
            return float("inf")  # Penalize failed trials

    # Run Optuna optimization
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)

    # Fit and return the final model using best parameters
    best = study.best_params
    final_model = SARIMAX(
        train,
        order=(best["p"], best["d"], best["q"]),
        seasonal_order=(best["P"], best["D"], best["Q"], seasonal_period),
        enforce_stationarity=False,
        enforce_invertibility=False,
    ).fit(disp=False)

    return final_model.forecast(steps=len(test))
