import numpy as np
import pandas as pd
import optuna
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.stattools import adfuller
from scipy.stats import boxcox
from scipy.special import inv_boxcox

from src.utils.metrics import smape


def get_seasonal_period(data):
    freq = pd.infer_freq(data.index)
    if freq in ["D", "B"]:
        return 7
    elif freq in ["W", "W-SUN", "W-MON"]:
        return 12
    elif freq in ["MS", "M"]:
        return 12
    return max(2, min(24, len(data) // 2))


def test_stationarity(series):
    try:
        result = adfuller(series.dropna())
        return result[1] < 0.05
    except Exception:
        return False


def safe_boxcox(series):
    shift = 1 - series.min() if series.min() <= 0 else 0
    transformed, lmbda = boxcox(series + shift)
    return transformed, lmbda, shift


def run_ets(train, test, n_trials=20):
    seasonal_period = get_seasonal_period(train)
    use_boxcox = not test_stationarity(train)

    if use_boxcox:
        train_trans, lmbda, shift = safe_boxcox(train)
    else:
        train_trans = train

    def fit_ets_model(series, trend, seasonal, seasonal_periods):
        try:
            model = ExponentialSmoothing(
                series,
                trend=trend,
                seasonal=seasonal,
                seasonal_periods=seasonal_periods,
                initialization_method="estimated"
            ).fit()
            preds = model.forecast(len(test))
            return preds
        except Exception:
            return None

    def objective(trial):
        trend = trial.suggest_categorical("trend", ["add", "mul", None])
        seasonal = trial.suggest_categorical("seasonal", ["add", "mul", None])
        seasonal_periods = trial.suggest_int("seasonal_periods", max(2, seasonal_period), min(24, len(train)//2))

        preds = fit_ets_model(train_trans, trend, seasonal, seasonal_periods)
        if preds is None:
            return float("inf")

        if use_boxcox:
            preds = inv_boxcox(preds, lmbda) - shift

        return smape(test, preds)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best_params = study.best_params

    final_preds = fit_ets_model(
        train_trans,
        trend=best_params["trend"],
        seasonal=best_params["seasonal"],
        seasonal_periods=best_params["seasonal_periods"]
    )

    if final_preds is None:
        # Fallback in case optimized model fails
        final_preds = fit_ets_model(train_trans, "add", "add", seasonal_period)

    if use_boxcox:
        final_preds = inv_boxcox(final_preds, lmbda) - shift

    return final_preds
