import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Lasso, LinearRegression
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tsa.stattools import acf, pacf
import optuna
from scipy.stats import skew
from itertools import combinations
import sys
import warnings

from src.utils.logger import setup_logger
from src.models.evaluate_models import evaluate_ts_models

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.simplefilter("ignore")


MIN_MONTHLY_DATA_SPAN = 48
logger = setup_logger("MONTHLY")

# Load dataset

def load_and_parse_date(df, date_column="Month"):
    parsed = False
    formats_to_try = ["%Y-%m", "%y-%b", "%b-%y"]
    
    for fmt in formats_to_try:
        try:
            df[date_column] = pd.to_datetime(df[date_column], format=fmt)
            parsed = True
            break
        except ValueError:
            continue

    if not parsed:
        try:
            df[date_column] = pd.to_datetime(df[date_column])
        except Exception as e:
            logger.debug(f"❌ Unable to parse the {date_column} column. Please use consistent date format.")
            raise e
            
    df.set_index(date_column, inplace=True)
    logger.debug(f"✅ {date_column} column successfully parsed and set as index.")
    return df

def preprocess_data(df, target_column="Target", apply_log_transform=False):
    df = df.copy()
    if len(df) < MIN_MONTHLY_DATA_SPAN:
        raise ValueError(f"Insufficient data. Provide at least (~{MIN_MONTHLY_DATA_SPAN} periods) of data.")
    
    external_features = [col for col in df.columns if col != target_column]

    if external_features:
        logger.info(f"Filling NaNs in external features: {external_features}")
        df[external_features] = df[external_features].fillna(0)

    if df[target_column].isnull().all():
        raise ValueError("All target values are missing. Cannot proceed with preprocessing.")
    
    if df[target_column].isnull().sum() > 0:
        logger.warning("Missing values detected in target. Applying forward fill.")
        df[target_column].fillna(method="ffill", inplace=True)

        if df[target_column].isnull().sum() > 0:
            logger.warning("Forward fill incomplete. Applying linear interpolation.")
            df[target_column].interpolate(method="linear", inplace=True)

        logger.warning("Forecast accuracy may be affected due to missing value imputation.")
    
    skewness_value = skew(df[target_column])
    logger.info(f"Skewness of target: {skewness_value:.3f}")
    if apply_log_transform and abs(skewness_value) > 1:
        logger.warning("High skewness detected. Applying log1p transformation.")
        df[target_column] = np.log1p(df[target_column])
    else:
        logger.success("Skewness acceptable. No transformation applied.")

    Q1, Q3 = df[target_column].quantile([0.25, 0.75])
    IQR = Q3 - Q1
    lower_bound, upper_bound = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    outliers = df[(df[target_column] < lower_bound) | (df[target_column] > upper_bound)]

    if not outliers.empty:
        logger.warning(f"{len(outliers)} outliers detected. Replacing with median.")
        median_val = df[target_column].median()
        df.loc[df[target_column] < lower_bound, target_column] = median_val
        df.loc[df[target_column] > upper_bound, target_column] = median_val

    return df

def create_lag_features(df, target_column="Target", max_lag=12):
    for lag in range(1, max_lag + 1):
        df[f'lag_{lag}'] = df[target_column].shift(lag)
    df.dropna(inplace=True)
    return df

def select_lag_features(df, target_column="Target", max_lag=12, acf_threshold=0.2, pacf_threshold=0.2, vif_threshold=5.0):
    # Compute ACF and PACF
    acf_vals = acf(df[target_column], nlags=max_lag, fft=False)[1:]
    pacf_vals = pacf(df[target_column], nlags=max_lag)[1:]

    # Select lags based on threshold
    selected_lags = [
        lag for lag, (a, p) in enumerate(zip(acf_vals, pacf_vals), start=1)
        if abs(a) > acf_threshold or abs(p) > pacf_threshold
    ]

    # Fallback: include all if nothing meets threshold
    if not selected_lags:
        selected_lags = list(range(1, max_lag + 1))

    selected_features = [f"lag_{lag}" for lag in selected_lags]
    lag_df = df[selected_features].copy()

    # Drop NA to avoid issues in VIF calculation
    lag_df.dropna(inplace=True)

    # Iteratively remove features with high VIF
    while True:
        vif = pd.Series(
            [variance_inflation_factor(lag_df.values, i) for i in range(lag_df.shape[1])],
            index=lag_df.columns,
        )
        max_vif = vif.max()
        if max_vif > vif_threshold:
            drop_feature = vif.idxmax()
            selected_features.remove(drop_feature)
            lag_df.drop(columns=drop_feature, inplace=True)
        else:
            break

    return selected_features

def select_best_lags(train_features,test_features,train_target,test_target,model,
                     max_lags=5,early_stopping=True,verbose=False):
    best_rmse = float("inf")
    best_lags = None

    all_lags = [col for col in train_features.columns if col.startswith("lag_")]
    max_r = min(len(all_lags), max_lags)

    for r in range(1, max_r + 1):
        for lag_subset in combinations(all_lags, r):
            try:
                X_train = train_features[list(lag_subset)]
                X_test = test_features[list(lag_subset)]
                model.fit(X_train, train_target)
                predictions = model.predict(X_test)
                rmse = np.sqrt(mean_squared_error(test_target, predictions))

                if verbose:
                    print(f"Lags: {lag_subset} -> RMSE: {rmse:.4f}")

                if rmse < best_rmse:
                    best_rmse = rmse
                    best_lags = list(lag_subset)

                    if early_stopping and rmse == 0:
                        if verbose:
                            print("Early stopping: perfect prediction.")
                        return best_lags

            except Exception as e:
                if verbose:
                    print(f"Error with lags {lag_subset}: {e}")
                continue

    return best_lags if best_lags else all_lags

def apply_lasso_selection(df, selected_features, target_column="Target", alpha=0.1):
    X = df[selected_features]
    y = df[target_column]
    lasso = Lasso(alpha=alpha)
    lasso.fit(X, y)
    return [f for f, c in zip(selected_features, lasso.coef_) if c != 0]

def optimize_rolling_window(df, target_column="Target", min_window=3, max_window=36):
    best_window = min_window
    best_rmse = float("inf")

    train_size = int(len(df) * 0.8)
    train, test = df.iloc[:train_size], df.iloc[train_size:]
    test_values = test[target_column].values

    for window in range(min_window, max_window + 1):
        rolling_mean = train[target_column].rolling(window=window, min_periods=1).mean()
        forecast = rolling_mean.iloc[-len(test):].values

        # Pad forecast if it’s shorter than test due to indexing issues
        if len(forecast) < len(test_values):
            forecast = np.pad(forecast, (len(test_values) - len(forecast), 0), mode='edge')

        rmse = np.sqrt(mean_squared_error(test_values, forecast))
        if rmse < best_rmse:
            best_rmse = rmse
            best_window = window

    logger.success(f"Best rolling window found (Daily): {best_window} days")

    df[target_column] = df[target_column].rolling(window=best_window, min_periods=1).mean()
    return df

# === Workflow ===
def full_monthly_pipeline(df, target_column="Target"):
    df = load_and_parse_date(df)
    df = preprocess_data(df, target_column)
    df = create_lag_features(df, target_column)
    selected_features = select_lag_features(df, target_column)
    
    # Split for RMSE lag optimization
    split_idx = int(len(df) * 0.8)
    train_df, test_df = df.iloc[:split_idx], df.iloc[split_idx:]
    train_features, test_features = train_df[selected_features], test_df[selected_features]
    train_target, test_target = train_df[target_column], test_df[target_column]
    
    model = LinearRegression()
    selected_features = select_best_lags(train_features, test_features, train_target, test_target, model)
    
    # Final refinement using Lasso
    selected_features = apply_lasso_selection(df, selected_features, target_column)
    
    # Rolling window smoothing
    df = optimize_rolling_window(df, target_column)
    
    # Final split after rolling
    train_df, test_df = df.iloc[:split_idx], df.iloc[split_idx:]
    train_features, test_features = train_df[selected_features], test_df[selected_features]
    train_target, test_target = train_df[target_column], test_df[target_column]

    return df, train_features, test_features, train_target, test_target, selected_features



def compute_monthly_forecast(df_monthly, n_trials):
    df, X_train, X_test, y_train, y_test, features = full_monthly_pipeline(df_monthly)
    monthly_result = evaluate_ts_models(y_train, y_test, X_train, X_test, n_trials)
    y_test = dict(zip(y_test.index.strftime("%Y-%m-%d"), y_test.values.tolist()))
    monthly_result = {**monthly_result, "y_test": y_test, "freq": "MS"}
    return monthly_result
