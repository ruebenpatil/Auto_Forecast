import numpy as np
import pandas as pd
import optuna
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from scipy.stats import boxcox
from scipy.special import inv_boxcox
from src.utils.metrics import smape


def get_seasonal_period(data):
    inferred_freq = pd.infer_freq(data.index)
    if inferred_freq in ["D", "B"]:
        return 7
    elif inferred_freq in ["W", "W-SUN", "W-MON"]:
        return 12
    elif inferred_freq in ["MS", "M"]:
        return 12
    return max(2, min(24, len(data) // 2))


def safe_boxcox(series):
    shift = 1 - series.min() if series.min() <= 0 else 0
    transformed, lmbda = boxcox(series + shift)
    return transformed, lmbda, shift


def run_holt_winters(train, test, n_trials=20):
    train_series = train
    test_series = test

    seasonal_period = get_seasonal_period(train_series)
    use_boxcox = (train_series <= 0).any()

    if use_boxcox:
        train_trans, lmbda, shift = safe_boxcox(train_series)
    else:
        train_trans = train_series

    def fit_holt_winters_model(series, trend, seasonal, seasonal_periods, smoothing_level, smoothing_slope, smoothing_seasonal):
        try:
            model = ExponentialSmoothing(
                series,
                trend=trend,
                seasonal=seasonal,
                seasonal_periods=seasonal_periods,
                initialization_method="estimated"
            )
            fit_model = model.fit(
                smoothing_level=smoothing_level,
                smoothing_slope=smoothing_slope if trend is not None else None,
                smoothing_seasonal=smoothing_seasonal if seasonal is not None else None
            )
            preds = fit_model.forecast(len(test_series))
            return preds
        except Exception:
            return None

    def objective(trial):
        trend = trial.suggest_categorical("trend", ["add", "mul", None])
        seasonal = trial.suggest_categorical("seasonal", ["add", "mul", None])
        seasonal_periods = trial.suggest_int("seasonal_periods", max(2, seasonal_period), min(24, len(train)//2))
        smoothing_level = trial.suggest_float("smoothing_level", 0.01, 0.99)
        smoothing_slope = trial.suggest_float("smoothing_slope", 0.01, 0.99)
        smoothing_seasonal = trial.suggest_float("smoothing_seasonal", 0.01, 0.99)

        preds = fit_holt_winters_model(
            train_trans,
            trend,
            seasonal,
            seasonal_periods,
            smoothing_level,
            smoothing_slope,
            smoothing_seasonal
        )

        if preds is None:
            return float("inf")

        if use_boxcox:
            preds = inv_boxcox(preds, lmbda) - shift

        return smape(test_series.values, preds)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, timeout=60, show_progress_bar=True)

    best = study.best_params

    final_preds = fit_holt_winters_model(
        train_trans,
        trend=best["trend"],
        seasonal=best["seasonal"],
        seasonal_periods=best["seasonal_periods"],
        smoothing_level=best["smoothing_level"],
        smoothing_slope=best["smoothing_slope"],
        smoothing_seasonal=best["smoothing_seasonal"]
    )

    if final_preds is None:
        # Fallback to simple additive model
        final_preds = fit_holt_winters_model(train_trans, "add", "add", seasonal_period, 0.6, 0.2, 0.2)

    if use_boxcox:
        final_preds = inv_boxcox(final_preds, lmbda) - shift

    return final_preds
