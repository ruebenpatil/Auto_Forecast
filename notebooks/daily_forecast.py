import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Lasso, LinearRegression
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tsa.stattools import acf, pacf
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_absolute_percentage_error
import torch
import pytorch_lightning as pl
from pytorch_forecasting import TemporalFusionTransformer
from pytorch_forecasting.data import TimeSeriesDataSet
from pytorch_forecasting.metrics import SMAPE
import optuna
from pytorch_lightning.callbacks import EarlyStopping
import torch.nn as nn
import torch.optim as optim
from prophet import Prophet
from scipy.stats import skew
from itertools import combinations
import sys
import warnings

warnings.filterwarnings("ignore")

# Load dataset
file_path = r"D:\Projects\time_series\Auto_Forecast\data\vigorous_daily_90day_data.csv"
df = pd.read_csv(file_path)

# Try parsing daily date formats
parsed = False
formats_to_try = ["%Y-%m-%d", "%d-%m-%Y", "%m/%d/%Y"]

for fmt in formats_to_try:
    try:
        df["Date"] = pd.to_datetime(df["Date"], format=fmt)
        parsed = True
        break
    except ValueError:
        continue

if not parsed:
    try:
        df["Date"] = pd.to_datetime(df["Date"])  # fallback auto-detection
        parsed = True
    except Exception as e:
        print("❌ Unable to parse the 'Date' column. Please use consistent date format.")
        raise e

# Set index
df.set_index("Date", inplace=True)
print("✅ Date column successfully parsed and set as index.")

# Identify external features
target_column = "Target"
external_features = [col for col in df.columns if col != target_column]
df[external_features] = df[external_features].fillna(0)

# Ensure minimum 3 Month of daily data (~90 rows)
if len(df) < 90:
    print("❌ Insufficient data. Please provide at least 3 Months (~90 days) of data.")
    sys.exit()

# Handle missing target values
if df[target_column].isnull().sum() > 0:
    print("⚠️ Missing values detected. Applying forward fill (ffill).")
    df[target_column].fillna(method='ffill', inplace=True)
    if df[target_column].isnull().sum() > 0:
        print("⚠️ Some values are still missing. Applying linear interpolation.")
        df[target_column].interpolate(method='linear', inplace=True)
    print("⚠️ Forecast accuracy may be affected due to existence of missing values.")

# Skewness
skewness = skew(df[target_column])
print(f"Skewness: {skewness}")
if abs(skewness) > 1:
    print("⚠️ High skewness detected. Applying log transformation.")
    df[target_column] = np.log(df[target_column] + 1)
else:
    print("✅ Skewness is acceptable. No transformation applied.")

# Outlier detection using IQR
Q1, Q3 = df[target_column].quantile([0.25, 0.75])
IQR = Q3 - Q1
lower_bound, upper_bound = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
outliers = df[(df[target_column] < lower_bound) | (df[target_column] > upper_bound)]
if not outliers.empty:
    print("⚠️ Outliers detected. Replacing with median.")
    df.loc[df[target_column] < lower_bound, target_column] = df[target_column].median()
    df.loc[df[target_column] > upper_bound, target_column] = df[target_column].median()

# Lag features (daily)
for lag in range(1, 31):  # 30 daily lags
    df[f'lag_{lag}'] = df[target_column].shift(lag)
df.dropna(inplace=True)

# ACF/PACF-based selection
acf_values = acf(df[target_column], nlags=30)[1:]
pacf_values = pacf(df[target_column], nlags=30)[1:]
selected_lags = [i + 1 for i, (a, p) in enumerate(zip(acf_values, pacf_values)) if abs(a) > 0.2 or abs(p) > 0.2]

if not selected_lags:
    selected_lags = list(range(1, 31))

# Remove multicollinear lags via VIF
selected_features = [f'lag_{lag}' for lag in selected_lags]
while len(selected_features) > 1:
    vif_data = pd.DataFrame()
    vif_data['Feature'] = selected_features
    vif_data['VIF'] = [variance_inflation_factor(df[selected_features].values, i) for i in range(len(selected_features))]
    
    max_vif = vif_data['VIF'].max()
    if max_vif > 5:
        selected_features.remove(vif_data.loc[vif_data['VIF'].idxmax(), 'Feature'])
    else:
        break

# RMSE-based lag selection
def select_best_lags(train_features, test_features, train_target, test_target, model, max_lags=5):
    best_rmse, best_lags = float("inf"), None
    all_lags = [col for col in train_features.columns if col.startswith("lag_")]
    
    for r in range(1, min(len(all_lags), max_lags) + 1):
        for lag_subset in combinations(all_lags, r):
            try:
                train_subset = train_features[list(lag_subset)]
                test_subset = test_features[list(lag_subset)]
                model.fit(train_subset, train_target)
                predictions = model.predict(test_subset)
                rmse = np.sqrt(mean_squared_error(test_target, predictions))
                if rmse < best_rmse:
                    best_rmse, best_lags = rmse, list(lag_subset)
            except Exception:
                continue
    
    return best_lags if best_lags else all_lags

# Prepare train/test split
split_index = int(len(df) * 0.8)
train_df, test_df = df.iloc[:split_index], df.iloc[split_index:]
train_features = train_df[selected_features]
test_features = test_df[selected_features]
train_target = train_df[target_column]
test_target = test_df[target_column]

model = LinearRegression()
best_lags = select_best_lags(train_features, test_features, train_target, test_target, model)
selected_features = best_lags

# Final selection via Lasso
X = df[selected_features]
y = df[target_column]
lasso = Lasso(alpha=0.1)
lasso.fit(X, y)
selected_features = [f for f, c in zip(selected_features, lasso.coef_) if c != 0]

# Apply rolling statistics dynamically with RMSE optimization (Daily Version)
best_window = 3  # Default to 3 days
best_rmse = float("inf")
train_size = int(len(df) * 0.8)
train, test = df.iloc[:train_size], df.iloc[train_size:]

for window in range(3, 37):  # Test rolling windows from 3 to 36 days
    rolling_mean = train[target_column].rolling(window=window, min_periods=1).mean()
    rmse = np.sqrt(mean_squared_error(test[target_column], rolling_mean[-len(test):]))
    if rmse < best_rmse:
        best_rmse = rmse
        best_window = window

print(f"✅ Best rolling window found (Daily): {best_window} days")
df[target_column] = df[target_column].rolling(window=best_window, min_periods=1).mean()

# Train-test split (re-split after rolling)
train_size = int(len(df) * 0.8)
train, test = df.iloc[:train_size], df.iloc[train_size:]
train_features, test_features = train[selected_features], test[selected_features]
train_target, test_target = train[target_column], test[target_column]

# Evaluation metrics remain the same
def smape(actual, forecast):
    return 100 * np.mean(2 * np.abs(forecast - actual) / (np.abs(actual) + np.abs(forecast)))

def mean_forecast_bias(actual, forecast):
    return np.mean(forecast - actual)

def evaluate_model(actual, forecast):
    return smape(actual, forecast), mean_forecast_bias(actual, forecast)

# TFT model with Optuna remains the same
def run_tft(train_features, test_features, train_target, test_target):
    def objective(trial):
        units = trial.suggest_int('units', 32, 256)
        dropout = trial.suggest_float('dropout', 0.1, 0.5)
        
        model = Sequential([
            Dense(units, activation='relu', input_shape=(train_features.shape[1],)),
            Dropout(dropout),
            Dense(units // 2, activation='relu'),
            Dropout(dropout),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        model.fit(train_features, train_target, epochs=50, batch_size=16, verbose=0)
        preds = model.predict(test_features).flatten()
        return smape(test_target, preds)
    
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=20)
    
    best_params = study.best_params
    model = Sequential([
        Dense(best_params['units'], activation='relu', input_shape=(train_features.shape[1],)),
        Dropout(best_params['dropout']),
        Dense(best_params['units'] // 2, activation='relu'),
        Dropout(best_params['dropout']),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(train_features, train_target, epochs=50, batch_size=16, verbose=0)
    optimized_preds = model.predict(test_features).flatten()
    
    baseline_model = Sequential([
        Dense(128, activation='relu', input_shape=(train_features.shape[1],)),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(1)
    ])
    baseline_model.compile(optimizer='adam', loss='mse')
    baseline_model.fit(train_features, train_target, epochs=50, batch_size=16, verbose=0)
    baseline_preds = baseline_model.predict(test_features).flatten()
    
    smape_base, mfb_base = evaluate_model(test_target, baseline_preds)
    smape_opt, mfb_opt = evaluate_model(test_target, optimized_preds)
    
    return optimized_preds if smape_opt < smape_base and abs(mfb_opt) < abs(mfb_base) else baseline_preds

# LSTM daily version
def run_lstm(train_features, test_features, train_target, test_target):
    train_X = np.asarray(train_features).reshape(train_features.shape[0], train_features.shape[1], 1)
    test_X = np.asarray(test_features).reshape(test_features.shape[0], test_features.shape[1], 1)
    
    train_y = np.asarray(train_target).reshape(-1, 1)
    test_y = np.asarray(test_target).reshape(-1, 1)

    def objective(trial):
        n_units = trial.suggest_int("n_units", 16, 128)
        dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5)
        learning_rate = trial.suggest_loguniform("learning_rate", 1e-4, 1e-2)

        model = Sequential([
            LSTM(n_units, activation='relu', return_sequences=True, input_shape=(train_X.shape[1], 1)),
            Dropout(dropout_rate),
            LSTM(n_units, activation='relu'),
            Dense(1)
        ])
        model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')
        model.fit(train_X, train_y, epochs=50, batch_size=16, verbose=0)

        predictions = model.predict(test_X).flatten()
        return smape(test_y, predictions)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=20)

    best_trial = study.best_trial
    best_model = Sequential([
        LSTM(best_trial.params["n_units"], activation='relu', return_sequences=True, input_shape=(train_X.shape[1], 1)),
        Dropout(best_trial.params["dropout_rate"]),
        LSTM(best_trial.params["n_units"], activation='relu'),
        Dense(1)
    ])
    best_model.compile(optimizer=Adam(learning_rate=best_trial.params["learning_rate"]), loss='mse')
    best_model.fit(train_X, train_y, epochs=50, batch_size=16, verbose=0)
    final_predictions = best_model.predict(test_X).flatten()
    
    return final_predictions

# GRU daily version
def run_gru(train_features, test_features, train_target, test_target):
    train_X = np.asarray(train_features).reshape(train_features.shape[0], train_features.shape[1], 1)
    test_X = np.asarray(test_features).reshape(test_features.shape[0], test_features.shape[1], 1)

    train_y = np.asarray(train_target).reshape(-1, 1)
    test_y = np.asarray(test_target).reshape(-1, 1)

    def objective(trial):
        n_units = trial.suggest_int("n_units", 16, 128)
        dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5)
        learning_rate = trial.suggest_loguniform("learning_rate", 1e-4, 1e-2)

        model = Sequential([
            GRU(n_units, activation='relu', return_sequences=True, input_shape=(train_X.shape[1], 1)),
            Dropout(dropout_rate),
            GRU(n_units, activation='relu'),
            Dense(1)
        ])
        model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')
        model.fit(train_X, train_y, epochs=50, batch_size=16, verbose=0)

        predictions = model.predict(test_X).flatten()
        return smape(test_y, predictions)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=20)

    best_trial = study.best_trial
    best_model = Sequential([
        GRU(best_trial.params["n_units"], activation='relu', return_sequences=True, input_shape=(train_X.shape[1], 1)),
        Dropout(best_trial.params["dropout_rate"]),
        GRU(best_trial.params["n_units"], activation='relu'),
        Dense(1)
    ])
    best_model.compile(optimizer=Adam(learning_rate=best_trial.params["learning_rate"]), loss='mse')
    best_model.fit(train_X, train_y, epochs=50, batch_size=16, verbose=0)

    baseline_model = Sequential([
        GRU(50, activation='relu', return_sequences=True, input_shape=(train_X.shape[1], 1)),
        GRU(50, activation='relu'),
        Dense(1)
    ])
    baseline_model.compile(optimizer='adam', loss='mse')
    baseline_model.fit(train_X, train_y, epochs=50, batch_size=16, verbose=0)

    best_preds = best_model.predict(test_X).flatten()
    baseline_preds = baseline_model.predict(test_X).flatten()

    if smape(test_y, best_preds) < smape(test_y, baseline_preds) and abs(mean_forecast_bias(test_y, best_preds)) < abs(mean_forecast_bias(test_y, baseline_preds)):
        return best_preds
    else:
        return baseline_preds

# --- Metrics ---
def smape(y_true, y_pred):
    return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true)))

def mfb(y_true, y_pred):
    return 100 * np.mean((y_pred - y_true) / y_true)

def mean_forecast_bias(actual, forecast):
    return np.mean(actual - forecast)

# --- ARIMA (Daily) ---
def run_arima(train, test):
    best_loss, best_order = float("inf"), None
    lambda_mfb = 0.5
    
    for p in range(3):
        for d in range(2):
            for q in range(3):
                try:
                    model = ARIMA(train, order=(p, d, q)).fit()
                    pred = model.forecast(steps=len(test))
                    loss = smape(test, pred) + lambda_mfb * abs(mfb(test, pred))
                    if loss < best_loss:
                        best_loss, best_order = loss, (p, d, q)
                except:
                    continue
                    
    final_model = ARIMA(train, order=best_order).fit()
    return final_model.forecast(steps=len(test))

# --- SARIMA (Daily) ---
def run_sarima(train, test):
    best_score, best_order, best_seasonal_order = float("inf"), None, None
    for p in range(3):
        for d in range(2):
            for q in range(3):
                for P in range(2):
                    for D in range(2):
                        for Q in range(2):
                            try:
                                model = SARIMAX(train, order=(p, d, q), seasonal_order=(P, D, Q, 7)).fit()
                                pred = model.forecast(steps=len(test))
                                combined_score = smape(test, pred) + abs(mean_forecast_bias(test, pred))
                                if combined_score < best_score:
                                    best_score, best_order, best_seasonal_order = combined_score, (p, d, q), (P, D, Q, 7)
                            except:
                                continue
    return SARIMAX(train, order=best_order, seasonal_order=best_seasonal_order).fit().forecast(steps=len(test))

# --- ETS (Daily with Optuna) ---
def run_ets(train, test):
    def objective(trial):
        trend = trial.suggest_categorical("trend", ["add", "mul", None])
        seasonal = trial.suggest_categorical("seasonal", ["add", "mul", None])
        seasonal_periods = trial.suggest_int("seasonal_periods", 2, min(30, len(train) // 2))

        try:
            model = ExponentialSmoothing(train, trend=trend, seasonal=seasonal, seasonal_periods=seasonal_periods)
            preds = model.fit().forecast(len(test))
            return smape(test, preds)
        except:
            return float("inf")
        
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=20)

    best_params = study.best_params
    optimized_model = ExponentialSmoothing(train, trend=best_params["trend"], seasonal=best_params["seasonal"], 
                                           seasonal_periods=best_params["seasonal_periods"]).fit()
    optimized_preds = optimized_model.forecast(len(test))

    baseline_model = ExponentialSmoothing(train, trend="add", seasonal="add", seasonal_periods=min(7, len(train) // 2)).fit()
    baseline_preds = baseline_model.forecast(len(test))

    return optimized_preds if smape(test, optimized_preds) < smape(test, baseline_preds) else baseline_preds

# --- Holt-Winters (Daily) ---
def run_holt_winters(train, test):
    train_series = pd.Series(train)

    def objective(trial):
        trend = trial.suggest_categorical("trend", ["add", "mul", None])
        seasonal = trial.suggest_categorical("seasonal", ["add", "mul", None])
        seasonal_period = trial.suggest_int("seasonal_periods", 2, min(30, max(2, len(train) // 2)))
        smoothing_level = trial.suggest_float("smoothing_level", 0.01, 1.0)
        smoothing_slope = trial.suggest_float("smoothing_slope", 0.01, 1.0)
        smoothing_seasonal = trial.suggest_float("smoothing_seasonal", 0.01, 1.0)

        try:
            model = ExponentialSmoothing(train_series, trend=trend, seasonal=seasonal, seasonal_periods=seasonal_period)
            fit_model = model.fit(smoothing_level=smoothing_level, smoothing_slope=smoothing_slope, smoothing_seasonal=smoothing_seasonal)
            forecast = fit_model.forecast(len(test))
            return smape(test, forecast)
        except:
            return float("inf")

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=20, timeout=60)

    best_params = study.best_params
    best_model = ExponentialSmoothing(train_series, 
                                      trend=best_params["trend"], 
                                      seasonal=best_params["seasonal"], 
                                      seasonal_periods=best_params["seasonal_periods"]).fit(
                                        smoothing_level=best_params["smoothing_level"],
                                        smoothing_slope=best_params["smoothing_slope"],
                                        smoothing_seasonal=best_params["smoothing_seasonal"]
                                      )
    return best_model.forecast(len(test))

# --- ML Models (RandomForest, XGBoost, SVR, etc.) for Daily Data ---
def run_ml_model(model, train, test, train_features, test_features):
    model.fit(train_features, train)
    base_predictions = model.predict(test_features)
    base_smape = smape(test, base_predictions)

    def objective(trial):
        param_grid = {
            "RandomForestRegressor": {
                "n_estimators": trial.suggest_int("n_estimators", 50, 300, step=50),
                "max_depth": trial.suggest_int("max_depth", 3, 20),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
            },
            "XGBRegressor": {
                "n_estimators": trial.suggest_int("n_estimators", 50, 300, step=50),
                "max_depth": trial.suggest_int("max_depth", 3, 20),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            },
            "LGBMRegressor": {
                "n_estimators": trial.suggest_int("n_estimators", 50, 300, step=50),
                "max_depth": trial.suggest_int("max_depth", 3, 20),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            },
            "CatBoostRegressor": {
                "iterations": trial.suggest_int("iterations", 50, 300, step=50),
                "depth": trial.suggest_int("depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            },
            "SVR": {
                "C": trial.suggest_float("C", 0.1, 10),
                "epsilon": trial.suggest_float("epsilon", 0.01, 1),
                "kernel": trial.suggest_categorical("kernel", ["linear", "rbf", "poly"]),
            },
            "KNeighborsRegressor": {
                "n_neighbors": trial.suggest_int("n_neighbors", 2, 20),
                "weights": trial.suggest_categorical("weights", ["uniform", "distance"]),
            },
        }

        model_class = model.__class__.__name__
        tuned_model = model.__class__(**param_grid.get(model_class, {}))
        tuned_model.fit(train_features, train)
        predictions = tuned_model.predict(test_features)
        return smape(test, predictions)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=20, timeout=120)

    best_params = study.best_params
    tuned_model = model.__class__(**{k: v for k, v in best_params.items() if k in model.get_params()})
    tuned_model.fit(train_features, train)
    tuned_predictions = tuned_model.predict(test_features)
    
    return tuned_predictions if smape(test, tuned_predictions) < base_smape else base_predictions

# Model for Prophet (Daily Adaptation)
def run_prophet(train, test):
    import pandas as pd
    import optuna
    from prophet import Prophet

    # Copy and reset index 
    test_df = test.copy()

    if train_df.index.name == "Date":
        train_df.reset_index(inplace=True)
    if test_df.index.name == "Date":
        test_df.reset_index(inplace=True)

    # Rename columns for Prophet
    train_df.rename(columns={"Date": "ds", "Target": "y"}, inplace=True)
    test_df.rename(columns={"Date": "ds"}, inplace=True)

    # Ensure datetime format
    train_df["ds"] = pd.to_datetime(train_df["ds"])
    test_df["ds"] = pd.to_datetime(test_df["ds"])

    # Check for holiday regressor
    has_holiday = "Holiday" in train_df.columns
    if has_holiday:
        train_df["holiday"] = train_df["Holiday"].fillna(0)

    def objective(trial):
        # Hyperparameter space
        params = {
            "changepoint_prior_scale": trial.suggest_float("changepoint_prior_scale", 0.001, 0.5, log=True),
            "seasonality_prior_scale": trial.suggest_float("seasonality_prior_scale", 0.01, 10.0, log=True),
            "seasonality_mode": trial.suggest_categorical("seasonality_mode", ["additive", "multiplicative"])
        }

        # Initialize Prophet model
        model = Prophet(
            changepoint_prior_scale=params["changepoint_prior_scale"],
            seasonality_prior_scale=params["seasonality_prior_scale"],
            seasonality_mode=params["seasonality_mode"],
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=True
        )

        # Add regressors if applicable
        if has_holiday:
            model.add_regressor("holiday")

        # Fit model
        model.fit(train_df)

        # Create future dataframe
        future = model.make_future_dataframe(periods=len(test_df), freq="D")
        if has_holiday:
            holiday_df = train_df[["ds", "holiday"]]
            future = future.merge(holiday_df, on="ds", how="left").fillna(0)

        # Generate forecast
        forecast = model.predict(future)
        y_pred = forecast.loc[forecast["ds"].isin(test_df["ds"]), "yhat"].values

        # Handle misalignment
        if len(y_pred) != len(test_df):
            return float("inf")

        return smape(test["Target"].values, y_pred)

    # Optimize Prophet using Optuna
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=20, show_progress_bar=False)

    # Best model training
    best_params = study.best_params
    final_model = Prophet(
        changepoint_prior_scale=best_params["changepoint_prior_scale"],
        seasonality_prior_scale=best_params["seasonality_prior_scale"],
        seasonality_mode=best_params["seasonality_mode"],
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=True
    )

    if has_holiday:
        final_model.add_regressor("holiday")

    final_model.fit(train_df)

    future = final_model.make_future_dataframe(periods=len(test_df), freq="D")
    if has_holiday:
        holiday_df = train_df[["ds", "holiday"]]
        future = future.merge(holiday_df, on="ds", how="left").fillna(0)

    forecast = final_model.predict(future)
    return forecast.loc[forecast["ds"].isin(test_df["ds"]), "yhat"].values

  # Extract only test period forecasts

# Function to calculate SMAPE
def smape(actual, predicted):
    return 100 * np.mean(np.abs(predicted - actual) / ((np.abs(actual) + np.abs(predicted)) / 2))

# Function to calculate MFB
def mfb(actual, predicted):
    return np.sum(predicted - actual) / np.sum(actual)

# Function to evaluate models
def evaluate_models(train, test, train_features, test_features):
    models = {
        "ARIMA": run_arima(train, test),
        "SARIMA": run_sarima(train, test),
        "ETS": run_ets(train, test),
        "Holt-Winters": run_holt_winters(train, test),
        "Random Forest": run_ml_model(RandomForestRegressor(n_estimators=100), train, test, train_features, test_features),
        "XGBoost": run_ml_model(XGBRegressor(), train, test, train_features, test_features),
        "LightGBM": run_ml_model(LGBMRegressor(), train, test, train_features, test_features),
        "CatBoost": run_ml_model(CatBoostRegressor(verbose=0), train, test, train_features, test_features),
        "SVR": run_ml_model(SVR(), train, test, train_features, test_features),
        "KNN": run_ml_model(KNeighborsRegressor(), train, test, train_features, test_features),
        "LSTM": run_lstm(train_features.values.reshape(-1, train_features.shape[1], 1),test_features.values.reshape(-1, test_features.shape[1], 1),train.values,test.values).flatten(),
        "GRU":  run_gru(train_features.values.reshape(-1, train_features.shape[1], 1),test_features.values.reshape(-1, test_features.shape[1], 1),train.values,test.values).flatten(),
        "TFT": run_tft(train_features, test_features, train, test),
        "Prophet": run_prophet(train_features.join(train), test_features.join(test))
    }
    
    smape_scores = {model: smape(test, pred) for model, pred in models.items()}
    mfb_scores = {model: mfb(test, pred) for model, pred in models.items()}
    
    # Define thresholds
    best_smape = min(smape_scores.values())
    smape_threshold = 1.5 * best_smape
    selected_models = {model: pred for model, pred in models.items() if smape_scores[model] <= smape_threshold and abs(mfb_scores[model]) <= 0.1}  # Keep models with MFB within ±10%
    
    print("\n✅ Models Used in Simple Average:")
    for model in selected_models.keys():
        print(f"- {model}")
    
    # Compute Simple Average
    simple_avg_forecast = np.mean(list(selected_models.values()), axis=0)
    smape_simple_avg = smape(test, simple_avg_forecast)
    mfb_simple_avg = mfb(test, simple_avg_forecast)
    
    # Compute Weighted Average
    # Filter models that return array-like predictions (exclude dict-based like Prophet, SARIMA)
    array_based_models = {
        model: pred for model, pred in models.items()
        if isinstance(pred, np.ndarray)
    }
    
    # Recompute weights for array-based models only
    inverse_smape = {
        model: 1 / smape_scores[model] for model in array_based_models if smape_scores[model] > 0
    }
    total_weight = sum(inverse_smape.values())
    weights = {model: weight / total_weight for model, weight in inverse_smape.items()}
    
    # Compute weighted average
    weighted_avg_forecast = sum(weights[model] * array_based_models[model] for model in array_based_models)

    smape_weighted_avg = smape(test, weighted_avg_forecast)
    mfb_weighted_avg = mfb(test, weighted_avg_forecast)
    
    smape_scores["Simple Average"] = smape_simple_avg
    smape_scores["Weighted Average"] = smape_weighted_avg
    mfb_scores["Simple Average"] = mfb_simple_avg
    mfb_scores["Weighted Average"] = mfb_weighted_avg

    # Hybrid Model Selection
    print("\n🔄 Evaluating Hybrid Models...")
    hybrid_model = None

    if smape_scores["Weighted Average"] < best_smape:
        hybrid_model = "Weighted Average"
    else:
        best_stat_model = min(["ARIMA", "SARIMA", "ETS", "Holt-Winters"], key=lambda m: smape_scores.get(m, float("inf")), default=None)
        best_ml_model = min(["Random Forest", "XGBoost", "LightGBM", "CatBoost", "SVR", "KNN"], key=lambda m: smape_scores.get(m, float("inf")), default=None)
        best_dl_model = min(["LSTM", "GRU", "TFT"], key=lambda m: smape_scores.get(m, float("inf")), default=None)
    
        if best_stat_model and best_ml_model:
            hybrid_forecast = 0.5 * models[best_stat_model] + 0.5 * models[best_ml_model]
            smape_hybrid = smape(test, hybrid_forecast)
            print(f"📊 Hybrid {best_stat_model} + {best_ml_model}: SMAPE = {smape_hybrid:.4f}")
            if smape_hybrid < best_smape:
                hybrid_model = f"{best_stat_model} + {best_ml_model}"
                models[hybrid_model] = hybrid_forecast
                smape_scores[hybrid_model] = smape_hybrid
    
        elif best_ml_model and best_dl_model:
            hybrid_forecast = 0.5 * models[best_ml_model] + 0.5 * models[best_dl_model]
            smape_hybrid = smape(test, hybrid_forecast)
            print(f"📊 Hybrid {best_ml_model} + {best_dl_model}: SMAPE = {smape_hybrid:.4f}")
            if smape_hybrid < best_smape:
                hybrid_model = f"{best_ml_model} + {best_dl_model}"
                models[hybrid_model] = hybrid_forecast
                smape_scores[hybrid_model] = smape_hybrid
    
    # Select Best Model
    best_model = min(smape_scores, key=smape_scores.get)
    
    return models, smape_scores, mfb_scores, best_model, simple_avg_forecast, weighted_avg_forecast

# Run pipeline
if __name__ == "__main__":
    import numpy as np

    # Run model evaluation
    models, smape_scores, mfb_scores, best_model, simple_avg_forecast, weighted_avg_forecast = evaluate_models(
        train[target_column], test[target_column], train_features, test_features
    )

    # Display SMAPE scores
    print("\n🔹 SMAPE Scores:")
    for model, score in smape_scores.items():
        print(f"{model}: {score:.4f}")

    # Display MFB scores
    print("\n🔹 MFB Scores:")
    for model, score in mfb_scores.items():
        print(f"{model}: {score:.4f}")

    # Show best model
    # print(f"\n✅ Best Model: {best_model} (SMAPE: {smape_scores[best_model]:.4f}, MFB: {mfb_scores[best_model]:.4f})")

    # # Model selection (only model name input, no forecast horizon)
    # while True:
    #     selected_model = input("\nEnter the model name you want to use for forecasting (from the above list): ").strip()
    #     if selected_model in models or selected_model in ["Simple Average", "Weighted Average"]:
    #         break
    #     print("Invalid model selection. Please enter a valid model name from the list.")

    # # Automatically use full test set length
    # forecast_horizon = len(test)

    # # Choose the forecast model
    # chosen_forecast = None
    # future_forecast_values = None

    # # Get forecast based on model selection
    # if selected_model == "Simple Average":
    #     chosen_forecast = simple_avg_forecast
    #     future_forecast_values = chosen_forecast[-forecast_horizon:]

    # elif selected_model == "Weighted Average":
    #     chosen_forecast = weighted_avg_forecast
    #     future_forecast_values = chosen_forecast[-forecast_horizon:]

    # elif selected_model in models:
    #     model_output = models[selected_model]

    #     # Handle Prophet
    #     if selected_model == "Prophet":
    #         chosen_forecast = model_output['yhat'][:len(test)]
    #         future_forecast_values = model_output['yhat'][-forecast_horizon:]

    #     # Handle SARIMA/ARIMA which returns separate components
    #     elif isinstance(model_output, dict) and "test" in model_output and "future" in model_output:
    #         chosen_forecast = model_output["test"]
    #         future_forecast_values = model_output["future"]

    #     # Generic forecast array
    #     else:
    #         chosen_forecast = model_output
    #         future_forecast_values = chosen_forecast[-forecast_horizon:]

    # else:
    #     print("⚠️ Invalid model selection. Using best model.")
    #     chosen_forecast = models[best_model]
    #     future_forecast_values = chosen_forecast[-forecast_horizon:]

    # # Format test index
    # test.index = pd.to_datetime(test.index)
    # formatted_test_index = test.index.strftime('%Y-%m-%d')

    # # Compute residuals + confidence interval
    # residuals = test[target_column] - chosen_forecast
    # std_dev = np.std(residuals)
    # z_score = 1.96
    # lower_bound = chosen_forecast - (z_score * std_dev)
    # upper_bound = chosen_forecast + (z_score * std_dev)

    # # Forecast DataFrame for test
    # forecast_df = pd.DataFrame({
    #     "Date": formatted_test_index,
    #     "Actual": test[target_column].values,
    #     "Forecast": chosen_forecast,
    #     "Lower Bound": lower_bound,
    #     "Upper Bound": upper_bound
    # })

    # print("\n📊 Actual vs Forecast (Test Dataset) with Confidence Interval:")
    # print(forecast_df.to_string(index=False))

    # # Generate future forecast index
    # last_test_date = test.index[-1]
    # future_dates = pd.date_range(start=last_test_date + pd.Timedelta(days=1), periods=forecast_horizon, freq='D')
    # formatted_future_dates = future_dates.strftime('%Y-%m-%d')

    # # Compute CI for future forecast
    # future_lower_bound = future_forecast_values - (z_score * std_dev)
    # future_upper_bound = future_forecast_values + (z_score * std_dev)

    # # Future forecast DataFrame
    # future_forecast_df = pd.DataFrame({
    #     "Date": formatted_future_dates,
    #     "Forecast": future_forecast_values,
    #     "Lower Bound": future_lower_bound,
    #     "Upper Bound": future_upper_bound
    # })

    # # Plot Actual vs Forecast with CI
    # plt.figure(figsize=(12, 6))
    # plt.plot(formatted_test_index, test[target_column], label="Actual", marker="o", linestyle="-", color="blue")
    # plt.plot(formatted_test_index, chosen_forecast, label=f"Forecast ({selected_model})", linestyle="--", color="red")
    # plt.fill_between(formatted_test_index, lower_bound, upper_bound, color="red", alpha=0.2, label="95% Confidence Interval")

    # plt.xlabel("Date")
    # plt.ylabel("Target")
    # plt.title("Actual vs Forecasted Target with Confidence Interval")
    # plt.legend()
    # plt.grid()
    # plt.tight_layout()
    # plt.show()

    # # Print future forecast
    # print("\n" + "="*80)
    # print("🧠  RP&KP | AutoForecast Suite — Intelligent Time Series Predictions".center(60))
    # print("="*80)

    # print(f"\n📅 Future Forecast for the next {forecast_horizon} days using {selected_model}:")
    # print(future_forecast_df.to_string(index=False))
    # print("="*80)
