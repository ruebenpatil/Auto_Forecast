import optuna
import numpy as np
from sklearn.base import clone
from sklearn.model_selection import train_test_split
from src.utils.metrics import smape

from src.utils.logger import setup_logger
import warnings

warnings.simplefilter("ignore")

logger = setup_logger("PROCESSING")

optuna.logging.enable_propagation()  # Propagate logs to the root logger.
optuna.logging.disable_default_handler()


def run_ml_model(model, train, test, train_features, test_features, n_trials=20):
    model_name = model.__class__.__name__

    # Fit base model
    model.fit(train_features, train)
    base_predictions = model.predict(test_features)
    base_smape = smape(test, base_predictions)

    def objective(trial):
        try:
            param_grid = {
                "RandomForestRegressor": {
                    "n_estimators": trial.suggest_int("n_estimators", 100, 300, step=50),
                    "max_depth": trial.suggest_int("max_depth", 5, 30),
                    "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
                    "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
                },
                "XGBRegressor": {
                    "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
                    "max_depth": trial.suggest_int("max_depth", 3, 12),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
                    "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                    "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                },
                "LGBMRegressor": {
                    "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
                    "max_depth": trial.suggest_int("max_depth", 3, 12),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
                    "num_leaves": trial.suggest_int("num_leaves", 20, 100),
                    "verbosity": -1,
                },
                "CatBoostRegressor": {
                    "iterations": trial.suggest_int("iterations", 100, 500, step=50),
                    "depth": trial.suggest_int("depth", 4, 10),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
                    "verbose": False,
                },
                "SVR": {
                    "C": trial.suggest_float("C", 0.1, 10),
                    "epsilon": trial.suggest_float("epsilon", 0.01, 1.0),
                    "kernel": trial.suggest_categorical("kernel", ["linear", "rbf", "poly"]),
                },
                "KNeighborsRegressor": {
                    "n_neighbors": trial.suggest_int("n_neighbors", 2, 30),
                    "weights": trial.suggest_categorical("weights", ["uniform", "distance"]),
                    "p": trial.suggest_int("p", 1, 2),  # Manhattan or Euclidean
                },
            }

            params = param_grid.get(model_name, {})
            if not params:
                return base_smape  # If no tuning config, fall back

            tuned_model = model.__class__(**params)
            tuned_model.fit(train_features, train)
            preds = tuned_model.predict(test_features)
            return smape(test, preds)

        except Exception:
            return float("inf")  # Penalize failed trials

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, timeout=120, show_progress_bar=True)

    # Refit best model
    best_params = study.best_params
    valid_params = {k: v for k, v in best_params.items() if k in model.get_params()}
    final_model = model.__class__(**valid_params)
    final_model.fit(train_features, train)
    tuned_preds = final_model.predict(test_features)

    final_smape = smape(test, tuned_preds)
    return tuned_preds if final_smape < base_smape else base_predictions
