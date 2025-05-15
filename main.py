from fastapi import FastAPI, UploadFile, Form, File, Request, Query, HTTPException
from fastapi.responses import JSONResponse
from starlette.middleware.sessions import SessionMiddleware
import traceback

from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pathlib import Path

import pandas as pd
import numpy as np
from io import StringIO
import json
import uuid

from src.forecast import (
    compute_daily_forecast,
    compute_monthly_forecast,
    compute_weekly_forecast,
)
from src.utils.logger import setup_logger

WORKING_DIR = Path.cwd().resolve()
TEMPLATE_PATH = WORKING_DIR.joinpath("templates")
STATIC_PATH = WORKING_DIR.joinpath("static")
SESSION_DIR = WORKING_DIR.joinpath("session")

SESSION_DIR.mkdir(exist_ok=True)

logger = setup_logger(__name__)


app = FastAPI()
templates = Jinja2Templates(directory=str(TEMPLATE_PATH))
app.mount("/static", StaticFiles(directory=str(STATIC_PATH)), name="static")
app.add_middleware(SessionMiddleware, secret_key="your-very-secret-key")


@app.get("/", name="home")
@app.get("/index", name="index")
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/upload", name="upload")
async def upload(request: Request):
    return templates.TemplateResponse("upload.html", {"request": request})

@app.get("/instructions", name="instructions")
async def instructions(request: Request):
    return templates.TemplateResponse("instructions.html", {"request": request})


@app.post("/api/forecast")
async def forecast(
    request: Request, file: UploadFile = File(...), frequency: str = Form(...)
):
    try:
        logger.debug(
            f"=============================== {file.filename}========================================="
        )
        logger.debug(f"Received file: {file.filename}")

        content = await file.read()
        df = pd.read_csv(StringIO(content.decode("utf-8")))

        n_trails = 20
        if frequency.lower() == "daily":
            result = compute_daily_forecast(df, n_trails)
        elif frequency.lower() == "monthly":
            result = compute_monthly_forecast(df, n_trails)
        elif frequency.lower() == "weekly":
            result = compute_weekly_forecast(df, n_trails)
        else:
            raise ValueError(
                "Invalid frequency. Must be one of: daily, weekly, monthly."
            )

        smape_scores_dict = result.get("smape_scores")
        mfb_scores_dict = result.get("mfb_scores")
        common_list = set(smape_scores_dict) | set(mfb_scores_dict)
        smape_rows = []
        mfb_rows = []

        for model in common_list:
            if smape_scores_dict.get(model) is not None:
                smape_rows.append(
                    [model, float(round(smape_scores_dict.get(model), 3))]
                )
            if mfb_scores_dict.get(model) is not None:
                mfb_rows.append([model, float(round(mfb_scores_dict.get(model), 3))])

        tables = [
            {
                "headers": ["Model", "Value"],
                "rows": smape_rows,
                "title": "SMAPE Scores",
            },
            {
                "headers": ["Model", "Value"],
                "rows": mfb_rows,
                "title": "MFB Scores",
            },
        ]

        best_model = result.get("best_model")
        session_key = str(uuid.uuid4())

        session_data = {
            "predictions_dict": result.get("models"),
            "best_model": best_model,
            "simple_avg_forecast": result.get("simple_avg_forecast"),
            "weighted_avg_forecast": result.get("weighted_avg_forecast"),
            "y_test": result.get("y_test"),
            "freq": result.get("freq"),
        }

        with open(SESSION_DIR.joinpath(f"{session_key}.json"), "w") as f:
            json.dump(session_data, f, indent=4)

        logger.debug(f"Session data: {session_data}")

        return JSONResponse(
            content={
                "message": "Forecasting completed successfully",
                "tables": tables,
                "models": list(common_list),
                "session_key": session_key,
                "top_model_text": f"Best Model: {best_model}  (SMAPE: {smape_scores_dict[best_model]:.3f})",
            },
            status_code=200,
        )

    except Exception as e:
        error_trace = traceback.format_exc()
        logger.error(f"Error occurred: {str(e)}\n{error_trace}")
        return JSONResponse(
            status_code=500,
            content={
                "error": "Forecasting failed",
                "details": str(e),  # You can replace with a custom message if needed
            },
        )


@app.get("/api/forecast/{model_name}")
async def forecast_model(
    request: Request, model_name: str, session_key: str = Query(...)
):
    try:
        session_file_path = SESSION_DIR.joinpath(f"{session_key}.json")

        if not session_file_path.exists():
            raise FileNotFoundError(f"No session found with key: {session_key}")

        with open(session_file_path, "r") as f:
            session_data = json.load(f)

        logger.debug(f"--------------- {model_name} --------------------")
        logger.debug(f"Session data: {session_key}")
        logger.debug(f"Session Key: {session_key}")

        y_test = pd.Series(session_data.get("y_test"))
        y_test.index = pd.to_datetime(y_test.index)

        forecast_horizon = len(y_test)
        models = session_data.get("predictions_dict")
        best_model = session_data.get("best_model")

        # Choose forecast
        if model_name == "Simple Average":
            chosen_forecast = session_data.get("simple_avg_forecast")
        elif model_name == "Weighted Average":
            chosen_forecast = session_data.get("weighted_avg_forecast")
        elif model_name in models:
            model_output = models[model_name]
            if model_name == "Prophet":
                chosen_forecast = model_output["yhat"][:forecast_horizon]
            elif isinstance(model_output, dict) and "test" in model_output:
                chosen_forecast = model_output["test"]
            else:
                chosen_forecast = model_output
        else:
            logger.warning("Invalid model selection. Using best model.")
            chosen_forecast = models[best_model]

        formatted_test_index = y_test.index.strftime("%Y-%m-%d")

        residuals = y_test - chosen_forecast
        std_dev = np.std(residuals)
        z_score = 1.96
        lower_bound = chosen_forecast - (z_score * std_dev)
        upper_bound = chosen_forecast + (z_score * std_dev)

        forecast_df = pd.DataFrame(
            {
                "Date": formatted_test_index.tolist(),
                "Actual": np.round(y_test.values, 3).tolist(),
                "Forecast": np.round(chosen_forecast, 3).tolist(),
                "Lower Bound": np.round(lower_bound, 3).tolist(),
                "Upper Bound": np.round(upper_bound, 3).tolist(),
            }
        ).sort_values(by="Date")

        # Future forecast dates
        last_test_date = y_test.index[-1]
        date_frequency = session_data.get("freq")

        if date_frequency == "D":
            last_test_date += pd.Timedelta(days=1)
            future_dates = pd.date_range(start=last_test_date, periods=forecast_horizon, freq="D")
            date_format = "%d/%m"
        elif date_frequency == "MS":
            future_dates = pd.date_range(start=last_test_date, periods=forecast_horizon + 1, freq="MS")[1:]
            date_format = "%m/%y"
        elif date_frequency == "W-MON":
            future_dates = pd.date_range(start=last_test_date, periods=forecast_horizon + 1, freq="W-MON")[1:]
            date_format = "%d/%m"
        else:
            raise ValueError("Invalid date frequency format in session data.")

        formatted_future_dates = future_dates.strftime("%Y-%m-%d")
        future_forecast_values = chosen_forecast[-forecast_horizon:]
        future_lower_bound = future_forecast_values - (z_score * std_dev)
        future_upper_bound = future_forecast_values + (z_score * std_dev)

        future_forecast_df = pd.DataFrame(
            {
                "Date": formatted_future_dates,
                "Forecast": np.round(future_forecast_values, 3).tolist(),
                "Lower Bound": np.round(future_lower_bound, 3).tolist(),
                "Upper Bound": np.round(future_upper_bound, 3).tolist(),
            }
        ).sort_values(by="Date")

        tables = [
            {
                "headers": ["Date", "Actual", "Forecast", "Lower Bound", "Upper Bound"],
                "rows": forecast_df.values.tolist(),
                "title": "Actual vs Forecast with Confidence Interval",
                "type": "current",
                "date_format": date_format,
            },
            {
                "headers": ["Date", "Forecast", "Lower Bound", "Upper Bound"],
                "rows": future_forecast_df.values.tolist(),
                "title": f"Future Forecast ({model_name})",
                "type": "future",
                "date_format": date_format,
            },
        ]

        return JSONResponse(content={"tables": tables}, status_code=200)

    except Exception as e:
        logger.error(f"Error in forecast_model: {str(e)}\n{traceback.format_exc()}")
        return JSONResponse(
            status_code=500,
            content={"error": "Failed to generate forecast", "details": str(e)},
        )
