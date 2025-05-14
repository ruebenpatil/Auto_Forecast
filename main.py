from fastapi import FastAPI, UploadFile, Form, File
from fastapi.responses import JSONResponse
from fastapi.requests import Request
from starlette.middleware.sessions import SessionMiddleware

from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pathlib import Path

import pandas as pd 
import numpy as np
from io import StringIO
import time
import json
from rich import print

from src.daily_forecast import compute_daily_forecast
from src.monthly_forecast import compute_monthly_forecast
from src.weekly_forecast import compute_weekly_forecast
from src.utils.logger import setup_logger

WORKING_DIR = Path.cwd().resolve()
TEMPLATE_PATH = WORKING_DIR.joinpath("templates")
STATIC_PATH = WORKING_DIR.joinpath("static")


templates = Jinja2Templates(directory=str(TEMPLATE_PATH))

app = FastAPI()

app.mount("/static", StaticFiles(directory=str(STATIC_PATH)), name="static")

app.add_middleware(
    SessionMiddleware,
    secret_key="your-very-secret-key"  # Use a strong, secure key in production
)


@app.get("/", name="home")
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/index", name="index")
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/upload", name="upload")
async def upload(request: Request):
    return templates.TemplateResponse("upload.html", {"request": request})


@app.post("/api/forecast")
async def forecast(request:Request, file: UploadFile = File(...), frequency: str = Form(...)):
    start_time = time.time()
    content = await file.read()
    df = pd.read_csv(StringIO(content.decode("utf-8")))

    if frequency.lower() == "daily":
        result = compute_daily_forecast(df, 20)
    elif frequency.lower() == "monthly":
        result = compute_monthly_forecast(df, 20)
    elif frequency.lower() == "weekly":
        result = compute_weekly_forecast(df,20)

    
    smape_scores_dict = result.get("smape_scores")
    mfb_scores_dict = result.get("mfb_scores")
    common_list = list(set(list(smape_scores_dict.keys())) & set(list(mfb_scores_dict.keys())))
    smape_rows = []
    mfb_rows = []
    
    for model in common_list:
        if smape_scores_dict.get(model) is not None:
            smape_rows.append([model, float(round(smape_scores_dict.get(model),3)) ])
        if mfb_scores_dict.get(model) is not None:
            mfb_rows.append([model, float(round(mfb_scores_dict.get(model),3)) ])

    tables= [
            {
                "headers": ["Model", "Value"],
                "rows":  smape_rows,
                "title": "SMAPE Scores",
            },
            {
                "headers": ["Model", "Value"],
                "rows":  mfb_rows,
                "title": "MFB Scores",
            },
        ]
    
    request.session.update({
            # "elapsed_time": (time.time() - start_time) / 60,
            # "results_ready": True,
            "predictions_dict": result.get("models"),
            # "smape_scores": result.get("smape_scores"),
            # "mfb_scores": result.get("mfb_scores"),
            # "best_model": result.get("best_model"),
            "simple_avg_forecast": result.get("simple_avg_forecast"),
            "weighted_avg_forecast": result.get("weighted_avg_forecast"),
            "y_test": result.get("y_test"),
            "freq": result.get("freq"),
        })
    
    print(request.session)
    print("----------------------------------------------------")
    print(tables)
    
    return JSONResponse(
        content={
            "message": "Forecasting completed successfully",
            "tables": tables,
            "models": common_list
        },
        status_code=200,
    )

@app.get("/api/forecast/{model_name}")
async def forecast_model(model_name: str, request: Request):
    print("----------------------------------------------------")
    print(request.session)
    print("----------------------------------------------------")
    print(request.session.get("weighted_avg_forecast"))
    y_test = request.session.get("y_test")
    y_test = pd.Series(y_test)
    y_test.index = pd.to_datetime(y_test.index)

    forecast_horizon = len(y_test)
    models = request.session.get("predictions_dict")

    if model_name == "Simple Average":
        chosen_forecast = request.session.get("simple_avg_forecast")
    elif model_name == "Weighted Average":
        chosen_forecast = request.session.get("weighted_avg_forecast")
    elif model_name in models:
        model_output = models[model_name]
        if model_name == "Prophet":
            chosen_forecast = model_output["yhat"][:forecast_horizon]
        elif isinstance(model_output, dict) and "test" in model_output:
            chosen_forecast = model_output["test"]
        else:
            chosen_forecast = model_output
    else:
        print("⚠️ Invalid model selection. Using best model.")
        chosen_forecast = models[model_name]

    formatted_test_index = y_test.index.strftime("%Y-%m-%d")

    residuals = y_test - chosen_forecast
    std_dev = np.std(residuals)
    z_score = 1.96
    lower_bound = chosen_forecast - (z_score * std_dev)
    upper_bound = chosen_forecast + (z_score * std_dev)

    forecast_df = pd.DataFrame({
            "Date": formatted_test_index.tolist(),
            "Actual": np.round(y_test.values,3).tolist(),
            "Forecast": np.round(chosen_forecast,3).tolist(),
            "Lower Bound": np.round(lower_bound,3).tolist(),
            "Upper Bound": np.round(upper_bound,3).tolist(),
        })
    forecast_df = forecast_df.sort_values(by="Date")

    # Future forecast dates
    last_test_date = y_test.index[-1]
    date_frequency = request.session.get("freq")

    if date_frequency == "D":
        last_test_date = last_test_date + pd.Timedelta(days=1)
        future_dates = pd.date_range(start=last_test_date,periods=forecast_horizon,freq="D")
        formatted_future_dates = future_dates.strftime("%Y-%m-%d")
        date_format = "%d/%m"
    elif date_frequency == "MS":
        future_dates = pd.date_range(start=last_test_date, periods=forecast_horizon + 1, freq="MS")[1:]
        formatted_future_dates = future_dates.strftime("%Y-%m-%d")
        date_format = "%m/%y"
    elif date_frequency == "W-MON":
        future_dates = pd.date_range(start=last_test_date, periods=forecast_horizon + 1, freq='W-MON')[1:]
        formatted_future_dates = future_dates.strftime('%Y-%m-%d')
        date_format = "%d/%m"

    future_forecast_values = chosen_forecast[-forecast_horizon:]
    future_lower_bound = future_forecast_values - (z_score * std_dev)
    future_upper_bound = future_forecast_values + (z_score * std_dev)

    future_forecast_df = pd.DataFrame({
        "Date": formatted_future_dates,
        "Forecast": np.round(future_forecast_values,3).tolist(),
        "Lower Bound": np.round(future_lower_bound,3).tolist(),
        "Upper Bound": np.round(future_upper_bound,3).tolist(),
    })
    future_forecast_df = future_forecast_df.sort_values(by="Date")

    tables = [
        {
            "headers": ["Date", "Actual", "Forecast", "Lower Bound", "Upper Bound"],
            "rows": forecast_df.values.tolist(),
            "title": "Actual vs Forecast with Confidence Interval",
            "type": "current",
            "date_format": date_format
        },
        {
            "headers": ["Date", "Forecast", "Lower Bound", "Upper Bound"],
            "rows": future_forecast_df.values.tolist(),
            "title": f"Future Forecast ({model_name})",
            "type": "future",
            "date_format": date_format
        },
    ]
    return JSONResponse(
        content = {
            "tables": tables,
        },
        status_code=200,
        )


