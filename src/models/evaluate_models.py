import numpy as np
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

from src.utils.logger import setup_logger
from src.utils.metrics import smape, mean_relative_forecast_bias

from src.models.tft_model import train_and_evaluate_tft_model
from src.models.lstm import train_lstm_model
from src.models.gru import train_and_evaluate_gru_model
from src.models.arima import run_arima, run_sarima
from src.models.ets import run_ets
from src.models.holt_winters import run_holt_winters
from src.models.ml_models import run_ml_model


logger = setup_logger(__name__)

def evaluate_ts_models(train, test, train_features, test_features, n_trials=10):
    models = {}
    smape_scores = {}
    mfb_scores = {}

    # --- Helper Functions ---
    def reshape_for_dl(X):
        return X.values.reshape(-1, X.shape[1], 1)

    def safe_run_model(name, func):
        try:
            pred = func()
            if isinstance(pred, dict):  # Skip incompatible formats
                raise ValueError("Prediction format not supported.")
            pred = np.array(pred).flatten()
            if pred.shape != test.shape:
                raise ValueError("Prediction shape mismatch.")
            models[name] = np.round(pred,3).tolist()
            smape_scores[name] = smape(test, pred)
            mfb_scores[name] = mean_relative_forecast_bias(test, pred)
        except Exception as e:
            print(f"[Skipping] {name}: {e}")

    # --- Model Definitions ---
    model_functions = {
        "ARIMA": lambda: run_arima(train, test),
        "SARIMA": lambda: run_sarima(train, test),
        "ETS": lambda: run_ets(train, test,n_trials),
        "Holt-Winters": lambda: run_holt_winters(train, test, n_trials),
        "Random Forest": lambda: run_ml_model(RandomForestRegressor(n_estimators=100), train, test, train_features, test_features,n_trials),
        "XGBoost": lambda: run_ml_model(XGBRegressor(verbose=False), train, test, train_features, test_features,n_trials),
        "LightGBM": lambda: run_ml_model(LGBMRegressor(), train, test, train_features, test_features,n_trials),
        "CatBoost": lambda: run_ml_model(CatBoostRegressor(verbose=False), train, test, train_features, test_features,n_trials),
        "SVR": lambda: run_ml_model(SVR(), train, test, train_features, test_features,n_trials),
        "KNN": lambda: run_ml_model(KNeighborsRegressor(), train, test, train_features, test_features,n_trials),
        "LSTM": lambda: train_lstm_model(reshape_for_dl(train_features), reshape_for_dl(test_features), train.values, test.values, n_trials).flatten(),
        "GRU": lambda: train_and_evaluate_gru_model(reshape_for_dl(train_features), reshape_for_dl(test_features), train.values, test.values, n_trials).flatten(),
        "TFT": lambda: train_and_evaluate_tft_model(train_features, test_features, train, test, n_trials),
    }

    # --- Run All Models ---
    for name, func in model_functions.items():
        print(f"Running {name}...")
        safe_run_model(name, func)

    if not smape_scores:
        raise RuntimeError("All models failed.")

    # --- Ensemble Thresholds ---
    best_smape = min(smape_scores.values())
    smape_threshold = 1.5 * best_smape

    selected_models = {
        name: pred for name, pred in models.items()
        if smape_scores.get(name, float("inf")) <= smape_threshold or abs(mfb_scores.get(name, float("inf"))) <= 0.1
    }

    logger.success("Models Used in Simple Average:")
    for model in selected_models:
        logger.debug(f"- {model}")

    # --- Simple Average ---
    if len(selected_models) > 0:
        simple_avg_forecast = np.mean(list(selected_models.values()), axis=0)
        smape_scores["Simple Average"] = smape(test, simple_avg_forecast)
        mfb_scores["Simple Average"] = mean_relative_forecast_bias(test, simple_avg_forecast)

    # --- Weighted Average ---
    array_models = {
        name: np.array(pred) for name, pred in models.items()
        if smape_scores.get(name, 0) > 0
    }
    if len(array_models) > 0:
        inverse_weights = {name: 1 / smape_scores[name] for name in array_models}
        total_weight = sum(inverse_weights.values())
        weights = {name: w / total_weight for name, w in inverse_weights.items()}

        weighted_avg_forecast = sum(weights[name] * array_models[name] for name in weights)
        weighted_avg_forecast = np.round(weighted_avg_forecast, 3).tolist()
        smape_scores["Weighted Average"] = smape(test, weighted_avg_forecast)
        mfb_scores["Weighted Average"] = mean_relative_forecast_bias(test, weighted_avg_forecast)

    # --- Hybrid Model ---
    hybrid_model = None
    if smape_scores.get("Weighted Average", float("inf")) < best_smape:
        hybrid_model = "Weighted Average"
    else:
        stat_models = ["ARIMA", "SARIMA", "ETS", "Holt-Winters"]
        ml_models = ["Random Forest", "XGBoost", "LightGBM", "CatBoost", "SVR", "KNN"]
        dl_models = ["LSTM", "GRU", "TFT"]

        best_stat = min(stat_models, key=lambda m: smape_scores.get(m, float("inf")), default=None)
        best_ml = min(ml_models, key=lambda m: smape_scores.get(m, float("inf")), default=None)
        best_dl = min(dl_models, key=lambda m: smape_scores.get(m, float("inf")), default=None)

        if best_stat and best_ml:
            hybrid_forecast = 0.5 * np.array(models[best_stat]) + 0.5 * np.array(models[best_ml])
            hybrid_smape = smape(test, hybrid_forecast)
            logger.info(f"Hybrid {best_stat} + {best_ml}: SMAPE = {hybrid_smape:.4f}")
            if hybrid_smape < best_smape:
                hybrid_model = f"{best_stat} + {best_ml}"
                models[hybrid_model] = np.round(hybrid_forecast,3).tolist()
                smape_scores[hybrid_model] = hybrid_smape

        elif best_ml and best_dl:
            hybrid_forecast = 0.5 * np.array(models[best_ml]) + 0.5 * np.array(models[best_dl])
            hybrid_smape = smape(test, hybrid_forecast)
            logger.info(f"Hybrid {best_ml} + {best_dl}: SMAPE = {hybrid_smape:.4f}")
            if hybrid_smape < best_smape:
                hybrid_model = f"{best_ml} + {best_dl}"
                models[hybrid_model] = np.round(hybrid_forecast,3).tolist()
                smape_scores[hybrid_model] = hybrid_smape

    # --- Final Selection ---
    best_model = min(smape_scores, key=smape_scores.get)

    return {
        "models": models,
        "smape_scores": smape_scores,
        "mfb_scores": mfb_scores,
        "best_model": best_model,
        "simple_avg_forecast": simple_avg_forecast.tolist(),
        "weighted_avg_forecast": weighted_avg_forecast,
    }
