from itertools import product
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from scipy.stats import boxcox
from scipy.special import inv_boxcox
import optuna

from src.utils.metrics import (
    mean_relative_forecast_bias,
    smape,
    mean_absolute_forecast_bias,
)


def get_seasonal_period(data):
    inferred_freq = pd.infer_freq(data.index)
    if inferred_freq in ["W", "W-MON", "W-SUN"]:
        return 12  # Weekly data
    elif inferred_freq in ["MS", "M"]:
        return 12  # Monthly data
    elif inferred_freq in ["D", "B"]:
        return 7  # Daily data
    else:
        return max(2, min(12, len(data) // 2))
    

def test_stationarity(series):
    result = adfuller(series.dropna())
    return result[1] < 0.05  # p-value

def safe_boxcox(series):
    shifted = series - series.min() + 1  # ensure all values > 0
    transformed, lmbda = boxcox(shifted)
    return transformed, lmbda, shifted.min()


def run_arima(train, test, n_trials=60, lambda_mfb=0.5):
    use_boxcox = not test_stationarity(train)
    use_boxcox = False
    if use_boxcox:
        train_trans, lmbda, shift = safe_boxcox(train)
    else:
        train_trans = train

    def objective(trial):
        p = trial.suggest_int("p", 0, 5)
        d = trial.suggest_int("d", 0, 2)
        q = trial.suggest_int("q", 0, 5)

        try:
            model = ARIMA(train_trans, order=(p, d, q)).fit()
            forecast = model.forecast(steps=len(test))

            if use_boxcox:
                forecast = inv_boxcox(forecast, lmbda) + shift - 1

            loss = smape(test, forecast) + lambda_mfb * abs(mean_relative_forecast_bias(test, forecast))
            return loss
        except Exception:
            return float("inf")

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best_params = study.best_params
    final_model = ARIMA(train_trans, order=(best_params["p"], best_params["d"], best_params["q"])).fit()
    final_forecast = final_model.forecast(steps=len(test))

    if use_boxcox:
        final_forecast = inv_boxcox(final_forecast, lmbda) + shift - 1

    return final_forecast


def run_sarima(train, test, n_trials=60):
    seasonal_period = get_seasonal_period(train)
    use_boxcox = not test_stationarity(train)
    use_boxcox = False
    if use_boxcox:
        train_trans, lmbda, shift = safe_boxcox(train)
    else:
        train_trans = train

    def objective(trial):
        p = trial.suggest_int("p", 0, 3)
        d = trial.suggest_int("d", 0, 2)
        q = trial.suggest_int("q", 0, 3)
        P = trial.suggest_int("P", 0, 2)
        D = trial.suggest_int("D", 0, 2)
        Q = trial.suggest_int("Q", 0, 2)

        try:
            model = SARIMAX(
                train_trans,
                order=(p, d, q),
                seasonal_order=(P, D, Q, seasonal_period),
                enforce_stationarity=False,
                enforce_invertibility=False,
            ).fit(disp=False)

            forecast = model.forecast(steps=len(test))

            if use_boxcox:
                forecast = inv_boxcox(forecast, lmbda) + shift - 1

            score = smape(test, forecast) + 0.5 * abs(mean_absolute_forecast_bias(test, forecast))
            return score
        except Exception:
            return float("inf")

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best = study.best_params
    final_model = SARIMAX(
        train_trans,
        order=(best["p"], best["d"], best["q"]),
        seasonal_order=(best["P"], best["D"], best["Q"], seasonal_period),
        enforce_stationarity=False,
        enforce_invertibility=False,
    ).fit(disp=False)

    final_forecast = final_model.forecast(steps=len(test))

    if use_boxcox:
        final_forecast = inv_boxcox(final_forecast, lmbda) + shift - 1

    return final_forecast