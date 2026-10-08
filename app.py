"""Single Streamlit dashboard for the academic flood prediction project."""
from pathlib import Path
import json
import math
import joblib
import pandas as pd
import plotly.express as px
import streamlit as st
from database import get_history, save_prediction, clear_history, MESSAGES
from train_model import FEATURES, SCENARIOS
from weather import CITIES, fetch_weather, weather_description

ROOT = Path(__file__).resolve().parent
COLORS = {"LOW": "#16a34a", "MEDIUM": "#eab308", "HIGH": "#dc2626"}
LIMITS = [(0., 300.), (0., 10.), (-10., 55.), (0., 100.)]
PAGES = ["Home Dashboard", "Flood Prediction", "Live Weather", "Prediction History", "About Project"]
st.set_page_config(page_title="FloodWatch • Smart Flood Prediction", page_icon="🌊", layout="wide")
st.markdown("""<style>
[data-testid="stMetric"] {background: white; border:1px solid #dce7f0;
 border-top:3px solid #0891b2; padding:18px; border-radius:12px;}
[data-testid="stSidebar"] {border-right:1px solid #dce7f0;}
.block-container {padding-top:2rem; padding-bottom:2rem; max-width:1250px;}
h1,h2,h3 {color:#10243a;} div.stButton > button {border-radius:9px;}
</style>""", unsafe_allow_html=True)


def validate(values):
    for name, value, (low, high) in zip(FEATURES, values, LIMITS):
        if not math.isfinite(value) or not low <= value <= high:
            raise ValueError(f"{name} must be between {low} and {high}.")


@st.cache_resource
def load_model():
    return joblib.load(ROOT / "flood_model.pkl")


def show_risk(risk):
    {"LOW": st.success, "MEDIUM": st.warning, "HIGH": st.error}[risk](f"{risk} RISK — {MESSAGES[risk]}")


def set_scenario(name):
    for feature, value in zip(FEATURES, SCENARIOS[name]):
        st.session_state[feature] = value
    st.session_state.source = f"Simulated scenario: {name}"
    st.session_state.pop("last_prediction", None)


def use_weather(city, current):
    # Current rain covers the API's reported interval, often 15 minutes.
    # Convert to an explicitly labeled equivalent hourly rate for this model.
    interval = current.get("interval", 900)
    st.session_state.rainfall = float(current["rain"]) * 3600 / interval
    st.session_state.temperature = float(current["temperature_2m"])
    st.session_state.humidity = float(current["relative_humidity_2m"])
    st.session_state.source = f"Open-Meteo {city}: equivalent hourly rainfall rate; manual water level"
    st.session_state.water_level = 0.0
    st.session_state.water_confirmed = False
    st.session_state.page = "Flood Prediction"
    st.session_state.pop("last_prediction", None)


def delete_history():
    clear_history()
    st.session_state.pop("last_prediction", None)


def rainfall_chart(history):
    chart = history.sort_values("id").copy()
    chart["timestamp"] = pd.to_datetime(chart.timestamp, utc=True)
    st.plotly_chart(px.line(chart, x="timestamp", y="rainfall", markers=True,
        labels={"rainfall": "Rainfall (mm per hour)", "timestamp": "Prediction time (UTC)"},
        color_discrete_sequence=["#0891b2"]), width="stretch")


for feature, value in zip(FEATURES, SCENARIOS["Normal Conditions"]):
    st.session_state.setdefault(feature, value)
st.session_state.setdefault("source", "Manual measurements")
st.sidebar.title("🌊 FloodWatch")
st.sidebar.caption("SMART FLOOD PREDICTION")
page = st.sidebar.radio("Navigation", PAGES, key="page")
st.sidebar.info("Academic proof of concept\n\nTrained on SIMULATED DATA. Follow official advisories for real safety decisions.")
st.title("AI-Based Smart Flood Prediction")
st.caption("Environmental measurements → machine learning → risk classification → alerts")
history = get_history()

if page == "Home Dashboard":
    st.subheader("Your conditions at a glance")
    if history.empty:
        st.info("Welcome! Make your first prediction on the Flood Prediction page to populate this dashboard.")
        values = [None] * 4
    else:
        values = history.iloc[0][FEATURES].tolist()
    for column, label, value in zip(st.columns(4),
            ["Rainfall (mm per hour)", "Water Level (m)", "Temperature (°C)", "Humidity (%)"], values):
        column.metric(label, "—" if value is None else f"{value:.1f}")
    if not history.empty:
        st.caption(f"Latest saved prediction • {history.iloc[0].timestamp} • {history.iloc[0].source}")
        show_risk(history.iloc[0].risk)
        st.subheader("Rainfall history")
        rainfall_chart(history)
        st.subheader("Recent predictions")
        st.dataframe(history.head(5), hide_index=True, width="stretch")

elif page == "Flood Prediction":
    st.subheader("Predict flood risk")
    st.caption("Rainfall is an hourly depth (mm). Water level uses an illustrative local reference.")
    for column, name in zip(st.columns(3), SCENARIOS):
        column.button(name, on_click=set_scenario, args=(name,), width="stretch")
    st.caption(f"Input source: {st.session_state.source}. Edited fields are user-entered overrides.")
    try:
        model = load_model()
    except (FileNotFoundError, ValueError, ImportError, EOFError) as error:
        st.error(f"Model cannot be loaded. Run python train_model.py. Details: {error}")
        st.stop()
    with st.form("prediction_form"):
        left, right = st.columns(2)
        left.number_input("Rainfall (mm per hour)", min_value=0., max_value=300., step=1., key="rainfall")
        right.number_input("Water level (m) — manually entered", min_value=0., max_value=10., step=0.1, key="water_level")
        left.number_input("Temperature (°C)", min_value=-10., max_value=55., step=0.5, key="temperature")
        right.number_input("Humidity (%)", min_value=0., max_value=100., step=1., key="humidity")
        if st.session_state.source.startswith("Open-Meteo"):
            st.checkbox("I have entered the local water level manually", key="water_confirmed")
        submitted = st.form_submit_button("Predict Flood Risk", type="primary", width="stretch")
    if submitted:
        try:
            values = [st.session_state[feature] for feature in FEATURES]
            validate(values)
            if st.session_state.source.startswith("Open-Meteo") and not st.session_state.get("water_confirmed"):
                raise ValueError("Enter and confirm the water level; Open-Meteo does not provide it.")
            risk = str(model.predict(pd.DataFrame([values], columns=FEATURES))[0])
            prediction_id = save_prediction(values, risk, st.session_state.source)
            st.session_state.last_prediction = (risk, prediction_id)
        except (ValueError, OSError) as error:
            st.error(str(error))
    if "last_prediction" in st.session_state:
        risk, prediction_id = st.session_state.last_prediction
        show_risk(risk)
        st.caption(f"Saved as prediction #{prediction_id}. View measurements and warnings in Prediction History.")
    st.info("Model predictions reproduce educational assumptions. Synthetic-data accuracy does not measure real-world forecasting ability.")

elif page == "Live Weather":
    st.subheader("Live weather • Open-Meteo")
    city = st.selectbox("Choose a city", list(CITIES))
    st.button("Refresh weather", type="primary")
    try:
        with st.spinner("Fetching real weather…"):
            current, forecast = fetch_weather(city)
        interval = current.get("interval", 900)
        rate = float(current["rain"]) * 3600 / interval
        for column, label, value in zip(st.columns(3),
                ["Temperature (°C)", "Humidity (%)", "Rain (mm in API interval)"],
                [current["temperature_2m"], current["relative_humidity_2m"], current["rain"]]):
            column.metric(label, value)
        st.caption(f"Real API data • {city} • {current['time']} IST • {weather_description(current['weather_code'])}")
        st.caption(f"Rain interval: {interval} seconds. Equivalent hourly rate: {rate:.2f} mm/h. This extrapolation is not an observed hourly total.")
        if not forecast.empty:
            st.plotly_chart(px.bar(forecast, x="time", y="rainfall",
                labels={"time": "Forecast time (IST)", "rainfall": "Hourly rain (mm)"},
                color_discrete_sequence=["#0891b2"]), width="stretch")
        st.button("Use Current Weather for Prediction", on_click=use_weather, args=(city, current),
                  disabled=not (0 <= rate <= 300 and -10 <= current["temperature_2m"] <= 55 and 0 <= current["relative_humidity_2m"] <= 100))
        st.info("Weather values will fill the form. You must enter and confirm water level manually.")
    except RuntimeError as error:
        st.error(str(error))
        st.info("Check your connection and allow api.open-meteo.com. Manual prediction remains available.")

elif page == "Prediction History":
    st.subheader("Saved predictions & alerts")
    st.caption("Timestamps are stored and displayed in UTC. Charts use actual saved predictions.")
    if history.empty:
        st.info("No predictions yet. Try a sample scenario on the Flood Prediction page.")
    else:
        st.dataframe(history, hide_index=True, width="stretch")
        distribution = history.risk.value_counts().rename_axis("risk").reset_index(name="count")
        st.plotly_chart(px.bar(distribution, x="risk", y="count", color="risk",
            color_discrete_map=COLORS, category_orders={"risk": ["LOW", "MEDIUM", "HIGH"]}), width="stretch")
        rainfall_chart(history)
        confirmed = st.checkbox("I confirm that all saved demo predictions should be deleted.")
        st.button("Clear demo history", disabled=not confirmed, on_click=delete_history)

else:
    st.subheader("About the project")
    st.markdown("""
**Abstract.** FloodWatch demonstrates how environmental measurements can pass through
an AI classifier to produce a clear risk indicator, dashboard and saved warning.

**Objectives.** Collect manual or real API weather inputs, classify risk, display alerts,
and preserve prediction history for review.

**Architecture.** Data Collection → Data Preprocessing → AI Prediction →
Risk Classification → Alert → Dashboard.

**Technologies.** Python, Streamlit, scikit-learn, Pandas, NumPy, Plotly, SQLite,
Requests and Joblib. No external database or paid API is needed.

**Algorithm.** A Random Forest combines many decision trees. Each tree learns patterns
from the simulated measurements; the forest votes for LOW, MEDIUM or HIGH. Raw numeric
columns are used in the same order at training and prediction; scaling is unnecessary for trees.

**Categories.** LOW: green status. MEDIUM: yellow caution. HIGH: red warning saved with the prediction.

**Benefits.** A simple, interactive demonstration of machine learning, API integration,
data persistence and visualization.

**Limitations.** Training labels are simulated educational assumptions, not scientific flood
thresholds. Water levels lack real river calibration. Weather is model-derived API data;
current rain is extrapolated to an hourly rate. There is no validated forecasting horizon,
sensor integration or emergency notification service. Use official local advisories.

**Future scope.** Collect verified local flood events, calibrate river measurements,
validate by location and time, add physical IoT sensors and SMS alerts.
""")
    metrics_path = ROOT / "data/model_metrics.json"
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text())
        st.subheader("Measured results • simulated holdout data")
        st.metric("Test accuracy", f"{metrics['accuracy']:.2%}")
        st.caption(f"{metrics['train_rows']} training rows / {metrics['test_rows']} test rows. Measures reproduction of simulated labels only.")
        st.dataframe(pd.DataFrame(metrics["classification_report"]).T, width="stretch")
        st.dataframe(pd.DataFrame(metrics["confusion_matrix"], index=metrics["labels"], columns=metrics["labels"]), width="stretch")
        st.caption("Confusion matrix: rows = actual labels; columns = predictions.")
