import streamlit as st
import pandas as pd
import time
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from src.daily_forecast import compute_daily_forecast
from src.monthly_forecast import compute_monthly_forecast
from src.weekly_forecast import compute_weekly_forecast
from src.utils.logger import setup_logger


logger = setup_logger("FORECAST")

# Page setup
st.set_page_config(page_title="Time Series Forecasting", layout="wide")
st.title("📈 Time Series Forecasting App")

# File upload
uploaded_file = st.file_uploader("Upload a CSV file with a datetime column", type="csv")
logger.debug("====================================================================================================")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    freq_option = st.selectbox("Select frequency", ["Daily", "Monthly", "Weekly"])

    if st.button("Run Forecast"):
        start_time = time.time()
        with st.spinner("Running forecast models..."):
            if freq_option == "Daily":
                result = compute_daily_forecast(df)
            elif freq_option == "Monthly":
                result = compute_monthly_forecast(df)
            elif freq_option == "Weekly":
                result = compute_weekly_forecast(df)

        st.session_state.update({
            "elapsed_time": (time.time() - start_time) / 60,
            "results_ready": True,
            "models": result.get("models"),
            "smape_scores": result.get("smape_scores"),
            "mfb_scores": result.get("mfb_scores"),
            "best_model": result.get("best_model"),
            "simple_avg_forecast": result.get("simple_avg_forecast"),
            "weighted_avg_forecast": result.get("weighted_avg_forecast"),
            "y_test": result.get("y_test"),
            "freq": result.get("freq"),
        })

    if st.session_state.get("results_ready", False):
        smape_df = pd.DataFrame(
            list(st.session_state["smape_scores"].items()), columns=["Model", "Value"]
        )
        mfb_df = pd.DataFrame(
            list(st.session_state["mfb_scores"].items()), columns=["Model", "Value"]
        )

        st.success(f"Forecasting completed in {st.session_state['elapsed_time']:.2f} minutes.")
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("SMAPE Scores")
            st.dataframe(smape_df, hide_index=True)
        with col2:
            st.subheader("MFB Scores")
            st.dataframe(mfb_df, hide_index=True)

        best_model = st.session_state["best_model"]
        smape_scores = st.session_state["smape_scores"]
        mfb_scores = st.session_state["mfb_scores"]
        models = st.session_state["models"]


        st.subheader(
            f"\n✅ Best Model: {best_model} (SMAPE: {smape_scores[best_model]:.4f})"
        )

        selected_model = st.selectbox("Select a forecasting model", smape_df["Model"].tolist())
        st.info(f"You selected the {selected_model} model. Implement your training logic here.")

        y_test = st.session_state["y_test"]
        simple_avg_forecast = st.session_state["simple_avg_forecast"]
        weighted_avg_forecast = st.session_state["weighted_avg_forecast"]
        forecast_horizon = len(y_test)

        # Model selection
        if selected_model == "Simple Average":
            chosen_forecast = simple_avg_forecast
        elif selected_model == "Weighted Average":
            chosen_forecast = weighted_avg_forecast
        elif selected_model in models:
            model_output = models[selected_model]
            if selected_model == "Prophet":
                chosen_forecast = model_output["yhat"][:forecast_horizon]
            elif isinstance(model_output, dict) and "test" in model_output:
                chosen_forecast = model_output["test"]
            else:
                chosen_forecast = model_output
        else:
            logger.debug("⚠️ Invalid model selection. Using best model.")
            chosen_forecast = models[best_model]

        print(y_test)

        y_test.index = pd.to_datetime(y_test.index)
        formatted_test_index = y_test.index.strftime("%Y-%m-%d")

        residuals = y_test - chosen_forecast
        std_dev = np.std(residuals)
        z_score = 1.96
        lower_bound = chosen_forecast - (z_score * std_dev)
        upper_bound = chosen_forecast + (z_score * std_dev)


        forecast_df = pd.DataFrame({
            "Date": formatted_test_index.tolist(),
            "Actual": y_test.values,
            "Forecast": chosen_forecast,
            "Lower Bound": lower_bound,
            "Upper Bound": upper_bound,
        })
        forecast_df = forecast_df.sort_values(by="Date")

        # Future forecast dates
        last_test_date = y_test.index[-1]

        if st.session_state.get("freq") == "D":
            last_test_date = last_test_date + pd.Timedelta(days=1)
            future_dates = pd.date_range(start=last_test_date,periods=forecast_horizon,freq="D")
            formatted_future_dates = future_dates.strftime("%Y-%m-%d")
        elif st.session_state.get("freq") == "MS":
            future_dates = pd.date_range(start=last_test_date, periods=forecast_horizon + 1, freq="MS")[1:]
            formatted_future_dates = future_dates.strftime("%Y-%m-%d")
        elif st.session_state.get("freq") == "W-MON":
            future_dates = pd.date_range(start=last_test_date, periods=forecast_horizon + 1, freq='W-MON')[1:]
            formatted_future_dates = future_dates.strftime('%Y-%m-%d')
        # Future values
        future_forecast_values = chosen_forecast[-forecast_horizon:]
        future_lower_bound = future_forecast_values - (z_score * std_dev)
        future_upper_bound = future_forecast_values + (z_score * std_dev)

        future_forecast_df = pd.DataFrame({
            "Date": formatted_future_dates,
            "Forecast": future_forecast_values,
            "Lower Bound": future_lower_bound,
            "Upper Bound": future_upper_bound,
        })
        future_forecast_df = future_forecast_df.sort_values(by="Date")

        # Display forecast and future forecast side by side
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📊 Actual vs Forecast (Test Dataset)")
            st.dataframe(forecast_df, hide_index=True)
            st.markdown("---")
            st.subheader("📈 Actual vs Forecast with Confidence Interval")
            fig1 = go.Figure()
            fig1.add_trace(go.Scatter(x=forecast_df["Date"], y=forecast_df["Actual"],
                                    mode='lines+markers', name='Actual'))
            fig1.add_trace(go.Scatter(x=forecast_df["Date"], y=forecast_df["Forecast"],
                                    mode='lines+markers', name='Forecast'))
            fig1.add_trace(go.Scatter(x=forecast_df["Date"], y=forecast_df["Upper Bound"],
                                    mode='lines', name='Upper Bound',
                                    line=dict(width=0), showlegend=False))
            fig1.add_trace(go.Scatter(x=forecast_df["Date"], y=forecast_df["Lower Bound"],
                                    fill='tonexty', mode='lines', name='Lower Bound',
                                    line=dict(width=0), fillcolor='rgba(255, 0, 0, 0.2)',
                                    showlegend=True))
            fig1.update_layout(title="Actual vs Forecasted with 95% CI",
                               legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5),
                               margin=dict(l=0, r=30, t=60, b=0),)
            st.plotly_chart(fig1, use_container_width=True)

        with col2:
            st.subheader(f"📅 Future Forecast ({selected_model})")
            st.dataframe(future_forecast_df, hide_index=True)
            st.markdown("---")
            st.subheader(f"📅 Future Forecast ({selected_model})")
            fig2 = go.Figure()
            x_complete = pd.to_datetime([*forecast_df["Date"].tolist(), *future_forecast_df["Date"]])
            y_complete = [*forecast_df["Actual"].tolist(),  *future_forecast_df["Forecast"]]
            fig2.add_trace(go.Scatter(x=x_complete, y=y_complete,
                                    mode='lines+markers', name="Overall"))
            fig2.add_trace(go.Scatter(x=forecast_df["Date"], y=forecast_df["Actual"],
                                    mode='lines+markers', name='Last Actual'))
            fig2.add_trace(go.Scatter(x=future_forecast_df["Date"], y=future_forecast_df["Forecast"],
                                    mode='lines+markers', name='Future Forecast'))
            fig2.add_trace(go.Scatter(x=future_forecast_df["Date"], y=future_forecast_df["Upper Bound"],
                                    mode='lines', name='Upper Bound', line=dict(width=0), showlegend=False))
            fig2.add_trace(go.Scatter(x=future_forecast_df["Date"], y=future_forecast_df["Lower Bound"],
                                    fill='tonexty', mode='lines', name='Lower Bound',
                                    line=dict(width=0), fillcolor='rgba(0, 255, 0, 0.2)', showlegend=True))
            fig2.update_layout(title="Future Forecast with 95% CI", 
                            legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5),
                               margin=dict(l=0, r=0, t=60, b=0),)
            st.plotly_chart(fig2, use_container_width=True)

        # Plotting
        # plt.figure(figsize=(12, 6))
        # plt.plot(formatted_test_index, y_test, label="Actual", marker="o", color="blue")
        # plt.plot(formatted_test_index, chosen_forecast, label=f"Forecast ({selected_model})", linestyle="--", color="red")
        # plt.fill_between(formatted_test_index, lower_bound, upper_bound, color="red", alpha=0.2, label="95% Confidence Interval")
        # plt.xlabel("Date")
        # plt.ylabel("Target")
        # plt.title("Actual vs Forecasted Target with Confidence Interval")
        # plt.legend()
        # plt.grid(True)
        # st.pyplot(plt)
