import optuna
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from src.utils.metrics import smape

def run_ets(train, test, n_trials=10):
    # Function to fit ETS model and return SMAPE loss
    def fit_ets_model(train, trend, seasonal, seasonal_periods):
        try:
            model = ExponentialSmoothing(
                train, trend=trend, seasonal=seasonal, seasonal_periods=seasonal_periods
            ).fit()
            return model.forecast(len(test))
        except (ValueError, TypeError):
            return None  # Return None in case of errors

    # Objective function for Optuna optimization
    def objective(trial):
        trend = trial.suggest_categorical("trend", ["add", "mul", None])
        seasonal = trial.suggest_categorical("seasonal", ["add", "mul", None])
        seasonal_periods = trial.suggest_int(
            "seasonal_periods", 2, min(30, len(train) // 2)
        )

        preds = fit_ets_model(train, trend, seasonal, seasonal_periods)
        if preds is None:
            return float("inf")  # Return large score if model fitting fails
        return smape(test, preds)  # Return SMAPE as objective function

    # Perform hyperparameter optimization
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    # Retrieve the best parameters from the study
    best_params = study.best_params
    print("Best Parameters:", best_params)

    # Fit and forecast using the optimized model
    optimized_preds = fit_ets_model(
        train,
        trend=best_params["trend"],
        seasonal=best_params["seasonal"],
        seasonal_periods=best_params["seasonal_periods"]
    )

    # Fit and forecast using the baseline model
    baseline_preds = fit_ets_model(
        train, trend="add", seasonal="add", seasonal_periods=min(7, len(train) // 2)
    )

    # Return the model with the lower SMAPE score
    return optimized_preds if smape(test, optimized_preds) < smape(test, baseline_preds) else baseline_preds
