import pandas as pd
from prophet import Prophet
import optuna

from src.utils.metrics import smape

def prepare_data(df, is_train=True):
    df = df.reset_index() if df.index.name == "Date" else df
    df["ds"] = pd.to_datetime(df["Date"])
    df = df.rename(columns={"Date": "ds"} if not is_train else {"Date": "ds", "Target": "y"})
    if "Holiday" in df.columns:
        df["holiday"] = df["Holiday"].fillna(0)
    return df

def build_model(params, use_holiday):
    model = Prophet(
        changepoint_prior_scale=params["changepoint_prior_scale"],
        seasonality_prior_scale=params["seasonality_prior_scale"],
        seasonality_mode=params["seasonality_mode"],
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=True,
    )
    if use_holiday:
        model.add_regressor("holiday")
    return model

def make_future_df(model, train_df, periods, use_holiday=False):
    future = model.make_future_dataframe(periods=periods, freq="D")
    if use_holiday:
        holiday_df = train_df[["ds", "holiday"]]
        future = future.merge(holiday_df, on="ds", how="left").fillna(0)
    return future

def run_prophet(train_df, test_df, n_trials=10):
    train_df = prepare_data(train_df, is_train=True)
    test_df = prepare_data(test_df, is_train=False)

    has_holiday = "holiday" in train_df.columns

    def objective(trial):
        params = {
            "changepoint_prior_scale": trial.suggest_float("changepoint_prior_scale", 0.001, 0.5, log=True),
            "seasonality_prior_scale": trial.suggest_float("seasonality_prior_scale", 0.01, 10.0, log=True),
            "seasonality_mode": trial.suggest_categorical("seasonality_mode", ["additive", "multiplicative"]),
        }

        model = build_model(params, has_holiday)
        model.fit(train_df)

        future = make_future_df(model, train_df, len(test_df), has_holiday)
        forecast = model.predict(future)
        y_pred = forecast.loc[forecast["ds"].isin(test_df["ds"]), "yhat"].values

        if len(y_pred) != len(test_df):
            return float("inf")

        return smape(test_df["Target"].values, y_pred)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best_model = build_model(study.best_params, has_holiday)
    best_model.fit(train_df)

    future = make_future_df(best_model, train_df, len(test_df), has_holiday)
    forecast = best_model.predict(future)

    return forecast.loc[forecast["ds"].isin(test_df["ds"]), "yhat"].values
