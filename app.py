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
from weather import CITIES, fetch_weather, weather_description, search_cities

ROOT = Path(__file__).resolve().parent
COLORS = {"LOW": "#398c74", "MEDIUM": "#c5a34e", "HIGH": "#c06a68"}
LIMITS = [(0., 300.), (0., 10.), (-10., 55.), (0., 100.)]
PAGES = ["Home Dashboard", "Flood Prediction", "Live Weather", "Prediction History", "About Project"]
st.set_page_config(page_title="FloodWatch | Environmental intelligence", page_icon="≈", layout="wide")
st.markdown(f"<style>{(ROOT / 'styles.css').read_text()}</style>", unsafe_allow_html=True)


def plot_chart(figure, height=300):
    """Apply the same quiet visual language to all data charts."""
    figure.update_layout(
        template="plotly_white", height=height, margin=dict(l=8, r=8, t=20, b=8),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Arial, sans-serif", size=12, color="#667c84"),
        hoverlabel=dict(bgcolor="#173943", font_color="white"),
        legend=dict(title_text="", orientation="h", y=1.15, x=0),
    )
    figure.update_xaxes(showgrid=False, zeroline=False, title_font_size=11)
    figure.update_yaxes(gridcolor="#e9efed", zeroline=False, title_font_size=11)
    st.plotly_chart(figure, width="stretch", config={"displayModeBar": False})


def display_history(records):
    """Use legible names in the dashboard; CSV exports retain all original columns."""
    view = records[["timestamp", "risk", "rainfall", "water_level", "temperature", "humidity"]].copy()
    view["timestamp"] = pd.to_datetime(view.timestamp, utc=True).dt.strftime("%d %b %Y · %H:%M")
    view = view.rename(columns={"timestamp": "Time (UTC)", "risk": "Risk", "rainfall": "Rain · mm/h",
                                "water_level": "Water · m", "temperature": "Temp · °C", "humidity": "Humidity · %"})
    st.dataframe(view, hide_index=True, width="stretch")


def go_to_prediction():
    st.session_state.page = "Flood Prediction"


def validate(values):
    for name, value, (low, high) in zip(FEATURES, values, LIMITS):
        if not math.isfinite(value) or not low <= value <= high:
            raise ValueError(f"{name} must be between {low} and {high}.")


@st.cache_resource
def load_model():
    return joblib.load(ROOT / "flood_model.pkl")


def show_risk(risk):
    palette = {"LOW": ("#23715d", "#edf6f1", "Low risk", "✓"),
               "MEDIUM": ("#97701b", "#fbf5e6", "Moderate risk", "!"),
               "HIGH": ("#b44747", "#fbeeee", "High risk", "!")}
    color, background, label, symbol = palette[risk]
    summary = "Review the warning below and follow official local advisories." if risk == "HIGH" else MESSAGES[risk]
    st.markdown(f"""<div class="status-panel" style="--status:{color};--status-bg:{background}">
      <div class="status-symbol">{symbol}</div><div><div class="status-label">Latest model assessment</div>
      <div class="status-heading">{label}</div><div class="status-copy">{summary}</div></div></div>""",
      unsafe_allow_html=True)
    if risk == "HIGH":
        st.error(MESSAGES[risk])


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
    plot_chart(px.area(chart, x="timestamp", y="rainfall", markers=True,
        labels={"rainfall": "Rainfall (mm per hour)", "timestamp": "Prediction time (UTC)"},
        color_discrete_sequence=["#117d79"]))


for feature, value in zip(FEATURES, SCENARIOS["Normal Conditions"]):
    st.session_state.setdefault(feature, value)
st.session_state.setdefault("source", "Manual measurements")
st.sidebar.markdown('<div class="brand"><div class="brand-mark">≈</div><div><div class="brand-name">FloodWatch</div><div class="brand-subtitle">Environmental intelligence</div></div></div>', unsafe_allow_html=True)
page = st.sidebar.radio("Workspace", PAGES, key="page", label_visibility="collapsed")
st.sidebar.markdown('<div class="sidebar-note"><strong>Academic demonstration</strong>Trained on simulated data.<br>For real safety decisions, follow official local advisories.</div>', unsafe_allow_html=True)
HEADINGS = {
    "Home Dashboard": ("Overview", "A clear view of your latest measurements and saved assessments."),
    "Flood Prediction": ("Assess flood risk", "Enter environmental measurements to run the trained model."),
    "Live Weather": ("Weather observatory", "Current conditions and rainfall forecasts, powered by Open-Meteo."),
    "Prediction History": ("Assessment history", "Review measurements, trace warnings and export your records."),
    "About Project": ("Behind the model", "Understand the method, measured results and limitations."),
}
st.markdown('<div class="page-top"><span class="eyebrow">FloodWatch / Workspace</span><span class="project-tag">Academic edition</span></div>', unsafe_allow_html=True)
title, description = HEADINGS[page]
st.title(title)
st.caption(description)
st.write("")
history = get_history()

if page == "Home Dashboard":
    values = [None] * 4 if history.empty else history.iloc[0][FEATURES].tolist()
    for column, label, value in zip(st.columns(4),
            ["Rainfall · mm/h", "Water level · m", "Temperature · °C", "Humidity · %"], values):
        column.metric(label, "—" if value is None else f"{value:.1f}")
    if history.empty:
        st.markdown('<div class="empty-state"><div class="eyebrow">Your workspace is ready</div><h3>Start with your first assessment</h3><p>Enter your measurements or try a sample scenario. Your results and charts will appear here.</p></div>', unsafe_allow_html=True)
        st.button("Create first assessment", type="primary", on_click=go_to_prediction)
    else:
        latest_time = pd.to_datetime(history.iloc[0].timestamp, utc=True).strftime("%d %b %Y · %H:%M UTC")
        st.caption(f"Latest saved readings · {latest_time}")
        show_risk(history.iloc[0].risk)
        trend, breakdown = st.columns([2, 1])
        with trend, st.container(border=True):
            st.subheader("Rainfall over time")
            st.caption("Measurements entered with each assessment")
            rainfall_chart(history)
        with breakdown, st.container(border=True):
            st.subheader("Risk distribution")
            st.caption(f"Across {len(history)} saved assessments")
            counts = history.risk.value_counts().rename_axis("risk").reset_index(name="count")
            figure = px.pie(counts, names="risk", values="count", hole=.72, color="risk", color_discrete_map=COLORS)
            figure.update_traces(textinfo="none", hovertemplate="%{label}: %{value} assessments<extra></extra>")
            figure.add_annotation(text=str(len(history)), x=.5, y=.5, showarrow=False, font_size=30, font_color="#173943")
            plot_chart(figure)
        with st.container(border=True):
            st.subheader("Recent assessments")
            display_history(history.head(5))
            st.caption("Showing the five most recent records. Full details and exports are in Prediction History.")

elif page == "Flood Prediction":
    st.subheader("Environmental inputs")
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
    st.caption("Academic model · Trained on simulated measurements. This assessment is not a validated flood forecast.")

elif page == "Live Weather":
    st.subheader("Choose a location")
    city = st.selectbox("Choose a city", list(CITIES))
    with st.expander("Search for another city"):
        with st.form("city_search"):
            query = st.text_input("City name", placeholder="For example: Pune")
            searched = st.form_submit_button("Search cities")
        if searched:
            st.session_state.city_results = []
            try:
                st.session_state.city_results = search_cities(query)
                if not st.session_state.city_results:
                    st.info("No cities found. Try a different spelling.")
            except (RuntimeError, ValueError) as error:
                st.error(str(error))
        results = st.session_state.get("city_results", [])
        if results:
            selected = st.selectbox("Matching locations", range(len(results)),
                                    format_func=lambda index: results[index]["label"])
            use_search = st.checkbox("Use this searched location", value=True)
            if use_search:
                city = results[selected]
    city_label = city["label"] if isinstance(city, dict) else city
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
        st.caption(f"Real API data • {city_label} • {current['time']} IST • {weather_description(current['weather_code'])}")
        st.caption(f"Rain interval: {interval} seconds. Equivalent hourly rate: {rate:.2f} mm/h. This extrapolation is not an observed hourly total.")
        if not forecast.empty:
            st.subheader("48-hour rainfall outlook")
            plot_chart(px.bar(forecast, x="time", y="rainfall",
                labels={"time": "Forecast time (IST)", "rainfall": "Hourly rain (mm)"},
                color_discrete_sequence=["#117d79"]))
        st.button("Use Current Weather for Prediction", on_click=use_weather, args=(city_label, current),
                  disabled=not (0 <= rate <= 300 and -10 <= current["temperature_2m"] <= 55 and 0 <= current["relative_humidity_2m"] <= 100))
        st.info("Weather values will fill the form. You must enter and confirm water level manually.")
    except RuntimeError as error:
        st.error(str(error))
        st.info("Check your connection and allow api.open-meteo.com. Manual prediction remains available.")

elif page == "Prediction History":
    st.subheader("Saved records")
    st.caption("Timestamps are stored and displayed in UTC. Charts use actual saved predictions.")
    if history.empty:
        st.info("No predictions yet. Try a sample scenario on the Flood Prediction page.")
    else:
        st.download_button("Download prediction history (CSV)",
                           data=history.to_csv(index=False).encode("utf-8-sig"),
                           file_name="flood_prediction_history.csv", mime="text/csv")
        display_history(history)
        with st.expander("Sources and saved warning messages"):
            st.dataframe(history[["id", "timestamp", "risk", "alert", "source"]], hide_index=True, width="stretch")
        distribution = history.risk.value_counts().rename_axis("risk").reset_index(name="count")
        st.subheader("Risk distribution")
        plot_chart(px.bar(distribution, x="risk", y="count", color="risk",
            color_discrete_map=COLORS, category_orders={"risk": ["LOW", "MEDIUM", "HIGH"]}))
        st.subheader("Rainfall over time")
        rainfall_chart(history)
        st.divider()
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
    st.subheader("What the model relies on")
    try:
        model = load_model()
        importance = pd.DataFrame({
            "Feature": ["Rainfall", "Water level", "Temperature", "Humidity"],
            "Importance (%)": model.feature_importances_ * 100,
        }).sort_values("Importance (%)")
        plot_chart(px.bar(importance, x="Importance (%)", y="Feature", orientation="h",
                              color_discrete_sequence=["#117d79"]))
        st.caption("These are the forest's overall training feature importances. They do not explain an individual prediction or prove real-world causes. They reflect simulated training labels.")
    except (FileNotFoundError, ValueError, ImportError, EOFError) as error:
        st.error(f"Feature importance is unavailable. Run python train_model.py. Details: {error}")
    metrics_path = ROOT / "data/model_metrics.json"
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text())
        st.subheader("Measured results • simulated holdout data")
        st.metric("Test accuracy", f"{metrics['accuracy']:.2%}")
        st.caption(f"{metrics['train_rows']} training rows / {metrics['test_rows']} test rows. Measures reproduction of simulated labels only.")
        st.dataframe(pd.DataFrame(metrics["classification_report"]).T, width="stretch")
        st.dataframe(pd.DataFrame(metrics["confusion_matrix"], index=metrics["labels"], columns=metrics["labels"]), width="stretch")
        st.caption("Confusion matrix: rows = actual labels; columns = predictions.")
