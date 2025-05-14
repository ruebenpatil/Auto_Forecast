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

MIN_WEEKLY_DATA_SPAN = 104

logger = setup_logger("WEEKLY")

def load_and_parse_date(df, date_column="Week"):
    parsed = False
    formats_to_try = ["%Y-%m-%d", "%d-%m-%Y", "%Y/%m/%d", "%d/%m/%Y"]
    
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
            logger.error("Unable to parse the 'Date' column. Please use consistent date format.")
            raise e
            
    df.set_index(date_column, inplace=True)
    logger.success("Date column successfully parsed and set as index.")
    return df

def preprocess_data(df, target_column="Sales"):
    if len(df) < MIN_WEEKLY_DATA_SPAN:
        logger.warning(f"Insufficient data. Provide at least (~{MIN_WEEKLY_DATA_SPAN} weeks) of data.")
        sys.exit()
    
    external_features = [col for col in df.columns if col != target_column]
    df[external_features] = df[external_features].fillna(0)
    
    if df[target_column].isnull().sum() > 0:
        logger.warning("Missing values detected. Applying forward fill (ffill).")
        df[target_column].fillna(method='ffill', inplace=True)
        if df[target_column].isnull().sum() > 0:
            logger.warning("Some values are still missing. Applying linear interpolation.")
            df[target_column].interpolate(method='linear', inplace=True)
        logger.warning("Forecast accuracy may be affected due to existence of missing values.")
    
    skewness_value = skew(df[target_column])
    logger.info(f"Skewness: {skewness_value}")
    if abs(skewness_value) > 1:
        logger.warning("High skewness detected. Applying log transformation.")
        df[target_column] = np.log(df[target_column] + 1)
    else:
        logger.success("Skewness is acceptable. No transformation applied.")
    
    Q1, Q3 = df[target_column].quantile([0.25, 0.75])
    IQR = Q3 - Q1
    lower_bound, upper_bound = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    outliers = df[(df[target_column] < lower_bound) | (df[target_column] > upper_bound)]
    
    if not outliers.empty:
        logger.warning("Outliers detected. Replacing with median.")
        df.loc[df[target_column] < lower_bound, target_column] = df[target_column].median()
        df.loc[df[target_column] > upper_bound, target_column] = df[target_column].median()
    
    return df

def create_lag_features(df, target_column="Sales", max_lag=52):
    for lag in range(1, max_lag + 1):
        df[f'lag_{lag}'] = df[target_column].shift(lag)
    df.dropna(inplace=True)
    return df

def select_lag_features(df, target_column="Sales"):
    max_lags = min(52, len(df) // 2 - 1)
    acf_vals = acf(df[target_column], nlags=max_lags)[1:]
    pacf_vals = pacf(df[target_column], nlags=max_lags)[1:]
    selected_lags = [i + 1 for i, (a, p) in enumerate(zip(acf_vals, pacf_vals)) if abs(a) > 0.2 or abs(p) > 0.2]
    
    if not selected_lags:
        selected_lags = list(range(1, 53))
    
    selected_features = [f'lag_{lag}' for lag in selected_lags]
    
    # Remove multicollinearity using VIF
    while len(selected_features) > 1:
        vif_data = pd.DataFrame()
        vif_data['Feature'] = selected_features
        vif_data['VIF'] = [variance_inflation_factor(df[selected_features].values, i) for i in range(len(selected_features))]
        
        max_vif = vif_data['VIF'].max()
        if max_vif > 5:
            selected_features.remove(vif_data.loc[vif_data['VIF'].idxmax(), 'Feature'])
        else:
            break

    return selected_features

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

def apply_lasso_selection(df, selected_features, target_column="Sales", alpha=0.1):
    X = df[selected_features]
    y = df[target_column]
    lasso = Lasso(alpha=alpha)
    lasso.fit(X, y)
    return [f for f, c in zip(selected_features, lasso.coef_) if c != 0]

def optimize_rolling_window(df, target_column="Sales", min_window=3, max_window=36):
    best_window = min_window
    best_rmse = float("inf")
    train_size = int(len(df) * 0.8)
    train, test = df.iloc[:train_size], df.iloc[train_size:]
    
    for window in range(min_window, max_window + 1):
        rolling_mean = train[target_column].rolling(window=window, min_periods=1).mean()
        rmse = np.sqrt(mean_squared_error(test[target_column], rolling_mean[-len(test):]))
        if rmse < best_rmse:
            best_rmse = rmse
            best_window = window

    logger.success(f"Best rolling window found (Daily): {best_window} days")
    df[target_column] = df[target_column].rolling(window=best_window, min_periods=1).mean()
    return df

# === Workflow ===
def full_weekly_pipeline(df_weekly, target_column="Sales"):
    df = load_and_parse_date(df_weekly)
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


# --------------------------
# Run Pipeline
# --------------------------
def compute_weekly_forecast(df_weekly, n_trials):
    df, X_train, X_test, y_train, y_test, features = full_weekly_pipeline(df_weekly)
    monthly_result = evaluate_ts_models(y_train, y_test, X_train, X_test, n_trials)
    y_test = dict(zip(y_test.index.strftime("%Y-%m-%d"), y_test.values.tolist()))
    monthly_result = {**monthly_result, "y_test": y_test, "freq": "W-MON"}
    return monthly_result