import warnings

from src.utils.logger import setup_logger
from src.models.evaluate_models import evaluate_ts_models
from src.processing.processing import full_daily_pipeline


warnings.simplefilter("ignore")


logger = setup_logger(__name__)


# Run pipeline
def compute_daily_forecast(df_daily, n_trials):
    formats_to_try = ["%Y-%m-%d", "%d-%m-%Y", "%m/%d/%Y"]
    target_column = "Target"
    date_column = "Date"
    min_daily_data_span = 90
    max_lags = 30
    df, X_train, X_test, y_train, y_test, features = full_daily_pipeline(
        df_daily,formats_to_try, target_column, date_column, min_daily_data_span, max_lags
    )
    daily_result = evaluate_ts_models(y_train, y_test, X_train, X_test, n_trials)
    y_test = dict(zip(y_test.index.strftime("%Y-%m-%d"), y_test.values.tolist()))
    daily_result = {**daily_result, "y_test": y_test, "freq": "D"}
    return daily_result

def compute_weekly_forecast(df_weekly, n_trials):
    formats_to_try = ["%Y-%m-%d", "%d-%m-%Y", "%m/%d/%Y"]
    target_column = "Target"
    date_column = "Week"
    min_weekly_data_span = 104
    max_lags = 52
    df, X_train, X_test, y_train, y_test, features = full_daily_pipeline(
        df_weekly, formats_to_try, target_column,date_column, min_weekly_data_span, max_lags
    )
    monthly_result = evaluate_ts_models(y_train, y_test, X_train, X_test, n_trials)
    y_test = dict(zip(y_test.index.strftime("%Y-%m-%d"), y_test.values.tolist()))
    monthly_result = {**monthly_result, "y_test": y_test, "freq": "W-MON"}
    return monthly_result


def compute_monthly_forecast(df_monthly, n_trials):
    formats_to_try = ["%Y-%m", "%y-%b", "%b-%y"]
    target_column = "Target"
    date_column = "Month"
    min_monthly_data_span = 48
    max_lags = 12
    df, X_train, X_test, y_train, y_test, features = full_daily_pipeline(
        df_monthly, formats_to_try, target_column,date_column, min_monthly_data_span, max_lags
    )
    monthly_result = evaluate_ts_models(y_train, y_test, X_train, X_test, n_trials)
    y_test = dict(zip(y_test.index.strftime("%Y-%m-%d"), y_test.values.tolist()))
    monthly_result = {**monthly_result, "y_test": y_test, "freq": "MS"}
    return monthly_result
