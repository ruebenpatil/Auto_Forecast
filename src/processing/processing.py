import pandas as pd
import numpy as np
from scipy.stats import skew
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import mean_squared_error
from itertools import combinations
from sklearn.linear_model import Lasso, LinearRegression
import optuna

from src.utils.logger import setup_logger
import warnings

warnings.simplefilter("ignore")

logger = setup_logger("PROCESSING")

optuna.logging.enable_propagation()  # Propagate logs to the root logger.
optuna.logging.disable_default_handler()


def load_and_parse_date(df, formats_to_try, date_column):
    df = df.copy()

    if df[date_column].isnull().any():
        logger.warning(
            "Null values found in date column. Consider handling them before parsing."
        )

    parsed = False

    for fmt in formats_to_try:
        try:
            df[date_column] = pd.to_datetime(
                df[date_column], format=fmt, errors="raise"
            )
            logger.info(f"Successfully parsed dates using format: {fmt}")
            parsed = True
            break
        except ValueError:
            continue

    if not parsed:
        try:
            df[date_column] = pd.to_datetime(
                df[date_column], infer_datetime_format=True, errors="raise"
            )
            logger.info("Parsed dates using fallback (inferred format).")
        except Exception as e:
            logger.error(
                "Unable to parse the 'Date' column. Please use a consistent format."
            )
    
    if not parsed:
        try:
            df[date_column] = pd.to_datetime(
                df[date_column], format="mixed", dayfirst=True
            )
            logger.info("Parsed dates using fallback (inferred format).")
        except Exception as e:
            logger.error(
                "Unable to parse the 'Date' column. Please use a consistent format."
            )
            raise e

    df.sort_values(by=date_column, inplace=True)
    df.set_index(date_column, inplace=True)

    if df.index.duplicated().any():
        logger.warning(
            "Duplicate dates found in the index. Consider aggregating or removing them."
        )

    logger.success("Date column successfully parsed and set as index.")
    return df


def preprocess_data(df, min_data_span, target_column, apply_log_transform=False):
    df = df.copy()
    if len(df) < min_data_span:
        raise ValueError(f"Insufficient data. Require at least ~{min_data_span} days.")

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

def create_lag_features(df, target_column, max_lags):
    for lag in range(1, max_lags + 1):
        df[f"lag_{lag}"] = df[target_column].shift(lag)
    df.dropna(inplace=True)
    return df

def select_lag_features(df, target_column, max_lags, acf_threshold=0.2, pacf_threshold=0.2, vif_threshold=5.0):
    # Compute ACF and PACF
    acf_vals = acf(df[target_column], nlags=max_lags, fft=False)[1:]
    pacf_vals = pacf(df[target_column], nlags=max_lags)[1:]

    # Select lags based on threshold
    selected_lags = [
        lag for lag, (a, p) in enumerate(zip(acf_vals, pacf_vals), start=1)
        if abs(a) > acf_threshold or abs(p) > pacf_threshold
    ]

    # Fallback: include all if nothing meets threshold
    if not selected_lags:
        selected_lags = list(range(1, max_lags + 1))

    selected_features = [f"lag_{lag}" for lag in selected_lags]
    lag_df = df[selected_features].copy()

    # Drop NA to avoid issues in VIF calculation
    lag_df.dropna(inplace=True)

    # Iteratively remove features with high VIF
    while len(selected_features) > 1:
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
                    logger.info(f"Lags: {lag_subset} -> RMSE: {rmse:.4f}")

                if rmse < best_rmse:
                    best_rmse = rmse
                    best_lags = list(lag_subset)

                    if early_stopping and rmse == 0:
                        if verbose:
                            logger.info("Early stopping: perfect prediction.")
                        return best_lags

            except Exception as e:
                if verbose:
                    logger.info(f"Error with lags {lag_subset}: {e}")
                continue

    return best_lags if best_lags else all_lags



def apply_lasso_selection(df, selected_features, target_column, alpha=0.1):
    X = df[selected_features]
    y = df[target_column]
    lasso = Lasso(alpha=alpha)
    lasso.fit(X, y)
    return [f for f, c in zip(selected_features, lasso.coef_) if c != 0]



def optimize_rolling_window(df, target_column, min_window=3, max_window=36):
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
def full_daily_pipeline(df, formats_to_try: list[str], target_column:str, date_column:str, min_data_span:int, max_lags:int):
    df = load_and_parse_date(df, formats_to_try, date_column)
    df = preprocess_data(df, min_data_span,target_column)
    df = create_lag_features(df, target_column, max_lags)
    selected_features = select_lag_features(df, target_column, max_lags)

    # Split for RMSE lag optimization
    split_idx = int(len(df) * 0.8)
    train_df, test_df = df.iloc[:split_idx], df.iloc[split_idx:]
    train_features, test_features = (
        train_df[selected_features],
        test_df[selected_features],
    )
    train_target, test_target = train_df[target_column], test_df[target_column]

    model = LinearRegression()
    best_lags = select_best_lags(
        train_features, test_features, train_target, test_target, model
    )

    # Final refinement using Lasso
    selected_features = apply_lasso_selection(df, best_lags, target_column)

    # Rolling window smoothing
    df = optimize_rolling_window(df, target_column)

    # Final split after rolling
    train_df, test_df = df.iloc[:split_idx], df.iloc[split_idx:]
    train_features, test_features = (
        train_df[selected_features],
        test_df[selected_features],
    )
    train_target, test_target = train_df[target_column], test_df[target_column]

    return (
        df,
        train_features,
        test_features,
        train_target,
        test_target,
        selected_features,
    )