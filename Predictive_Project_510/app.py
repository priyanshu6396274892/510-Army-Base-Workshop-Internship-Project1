"""
app.py
======
Enterprise Predictive Maintenance Dashboard (Streamlit).

- Reads the latest 50 rows of live telemetry from telemetry.db
- Engineers the same rolling-window features used at training time
- Scales + scores the latest reading with rul_model.pkl / scaler.pkl
- Displays a live health gauge, temperature/vibration trend charts,
  a prominent "Predicted Remaining Useful Life" countdown card, and
  a flashing red alert banner when predicted RUL drops below 15 hours.

Run:
    streamlit run app.py
"""

import time
from datetime import datetime

import numpy as np
import pandas as pd
import joblib
import plotly.graph_objects as go
import streamlit as st

import database_manager as db

# --------------------------------------------------------------------------
# Configuration (must match train_model.py)
# --------------------------------------------------------------------------
MODEL_PATH = "rul_model.pkl"
SCALER_PATH = "scaler.pkl"
FEATURE_COLUMNS_PATH = "feature_columns.pkl"
ROLLING_WINDOW = 5
ROLLING_SENSORS = ["temperature", "vibration", "pressure", "rotational_speed"]
ROWS_TO_FETCH = 50
CRITICAL_RUL_THRESHOLD = 15.0     # hours -- triggers red alert
WARNING_RUL_THRESHOLD = 40.0      # hours -- triggers amber caution
MAX_GAUGE_RUL = 125.0             # matches RUL_CLIP used at training time
REFRESH_SECONDS = 5

st.set_page_config(
    page_title="Predictive Maintenance | Machine Health",
    page_icon="🛠️",
    layout="wide",
)


# --------------------------------------------------------------------------
# Cached resource loaders
# --------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        feature_columns = joblib.load(FEATURE_COLUMNS_PATH)
        return model, scaler, feature_columns
    except FileNotFoundError:
        return None, None, None


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Recreates the rolling mean/std features exactly as done in train_model.py."""
    df = df.sort_values("cycle").copy()
    for sensor in ROLLING_SENSORS:
        df[f"{sensor}_rolling_mean_{ROLLING_WINDOW}"] = (
            df[sensor].rolling(ROLLING_WINDOW, min_periods=1).mean()
        )
        df[f"{sensor}_rolling_std_{ROLLING_WINDOW}"] = (
            df[sensor].rolling(ROLLING_WINDOW, min_periods=1).std().fillna(0.0)
        )
    return df


def predict_rul(df_featured: pd.DataFrame, model, scaler, feature_columns) -> np.ndarray:
    X = df_featured[feature_columns]
    X_scaled = scaler.transform(X)
    return model.predict(X_scaled)


def make_gauge(current_rul: float) -> go.Figure:
    if current_rul < CRITICAL_RUL_THRESHOLD:
        bar_color = "#E63946"
    elif current_rul < WARNING_RUL_THRESHOLD:
        bar_color = "#F4A261"
    else:
        bar_color = "#2A9D8F"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=current_rul,
        number={"suffix": " hrs", "font": {"size": 40}},
        title={"text": "Machine Health (Predicted RUL)", "font": {"size": 18}},
        gauge={
            "axis": {"range": [0, MAX_GAUGE_RUL], "tickwidth": 1},
            "bar": {"color": bar_color, "thickness": 0.3},
            "steps": [
                {"range": [0, CRITICAL_RUL_THRESHOLD], "color": "#F8D7DA"},
                {"range": [CRITICAL_RUL_THRESHOLD, WARNING_RUL_THRESHOLD], "color": "#FDEBD0"},
                {"range": [WARNING_RUL_THRESHOLD, MAX_GAUGE_RUL], "color": "#D4EFDF"},
            ],
            "threshold": {
                "line": {"color": "#E63946", "width": 4},
                "thickness": 0.85,
                "value": CRITICAL_RUL_THRESHOLD,
            },
        },
    ))
    fig.update_layout(height=320, margin=dict(l=20, r=20, t=60, b=10))
    return fig


# --------------------------------------------------------------------------
# Live dashboard fragment (auto-refreshes without reloading the whole page)
# --------------------------------------------------------------------------
@st.fragment(run_every=REFRESH_SECONDS)
def render_live_dashboard(selected_machine):
    model, scaler, feature_columns = load_artifacts()

    if model is None:
        st.error(
            "Model artifacts not found. Run `python train_model.py` first to generate "
            "`rul_model.pkl`, `scaler.pkl`, and `feature_columns.pkl`."
        )
        return

    rows = db.fetch_latest(ROWS_TO_FETCH) 

   
    import pandas as pd
    if isinstance(rows, pd.DataFrame) and not rows.empty:
        rows = rows[rows['machine_id'] == selected_machine]
    
    if not rows:
        st.warning(
            "No telemetry in the database yet. Start `mqtt_publisher.py` and "
            "`mqtt_subscriber.py` to begin streaming live sensor data."
        )
        return

    rows = db.fetch_latest(ROWS_TO_FETCH)
    raw_df = pd.DataFrame(rows)
    raw_df["timestamp"] = pd.to_datetime(raw_df["timestamp"])

    
    if not raw_df.empty and "machine_id" in raw_df.columns:
        raw_df = raw_df[raw_df["machine_id"] == selected_machine]
  

    featured_df = engineer_features(raw_df)
    predictions = predict_rul(featured_df, model, scaler, feature_columns)
    featured_df["predicted_rul_hours"] = np.clip(predictions, 0, None)
    latest = featured_df.iloc[-1]
    current_rul = float(latest["predicted_rul_hours"])
    machine_id = latest["machine_id"]
    last_update = latest["timestamp"]

    # ---------------- Alert banner ----------------
    if current_rul < CRITICAL_RUL_THRESHOLD:
        st.markdown(
            f"""
            <div style="background-color:#E63946; padding:16px; border-radius:10px;
                        text-align:center; animation: pulse 1.2s infinite;">
                <span style="color:white; font-size:22px; font-weight:700;">
                    🚨 CRITICAL ALERT — {machine_id} — Predicted RUL {current_rul:.1f} hrs — SCHEDULE IMMEDIATE MAINTENANCE
                </span>
            </div>
            <style>
            @keyframes pulse {{
                0%   {{ opacity: 1.0; }}
                50%  {{ opacity: 0.55; }}
                100% {{ opacity: 1.0; }}
            }}
            </style>
            """,
            unsafe_allow_html=True,
        )
        print(f"[ALERT] {datetime.now().isoformat()} - {machine_id} predicted RUL "
              f"{current_rul:.1f}h is BELOW critical threshold ({CRITICAL_RUL_THRESHOLD}h)!")
    elif current_rul < WARNING_RUL_THRESHOLD:
        st.markdown(
            f"""
            <div style="background-color:#F4A261; padding:12px; border-radius:10px; text-align:center;">
                <span style="color:#3B2400; font-size:18px; font-weight:600;">
                    ⚠️ Caution — {machine_id} — Predicted RUL {current_rul:.1f} hrs — Plan upcoming maintenance
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div style="background-color:#2A9D8F; padding:12px; border-radius:10px; text-align:center;">
                <span style="color:white; font-size:18px; font-weight:600;">
                    ✅ Nominal — {machine_id} — Predicted RUL {current_rul:.1f} hrs
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ---------------- Top row: gauge + countdown card + key metrics ----------------
    col_gauge, col_countdown, col_metrics = st.columns([1.2, 1, 1])

    with col_gauge:
        st.plotly_chart(make_gauge(current_rul), use_container_width=True, key="gauge_chart")

    with col_countdown:
        card_color = (
            "#E63946" if current_rul < CRITICAL_RUL_THRESHOLD
            else "#F4A261" if current_rul < WARNING_RUL_THRESHOLD
            else "#2A9D8F"
        )
        st.markdown(
            f"""
            <div style="background-color:{card_color}; padding:28px; border-radius:14px;
                        text-align:center; height:280px; display:flex; flex-direction:column;
                        justify-content:center;">
                <div style="color:white; font-size:16px; font-weight:500; letter-spacing:1px;">
                    PREDICTED REMAINING USEFUL LIFE
                </div>
                <div style="color:white; font-size:56px; font-weight:800; margin-top:10px;">
                    {current_rul:.1f}
                </div>
                <div style="color:white; font-size:20px; font-weight:600;">
                    HOURS
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_metrics:
        st.metric("Machine ID", machine_id)
        st.metric("Latest Cycle", int(latest["cycle"]))
        st.metric("Last Update (UTC)", last_update.strftime("%H:%M:%S"))
        st.metric("Rows in Live Window", len(featured_df))

    st.markdown("---")

    # ---------------- Trend charts ----------------
    st.subheader("📈 Live Sensor Trends")
    chart_df = featured_df.set_index("timestamp")

    col_temp, col_vib = st.columns(2)
    with col_temp:
        st.markdown("**Temperature**")
        st.line_chart(chart_df[["temperature"]], height=280)
    with col_vib:
        st.markdown("**Vibration**")
        st.line_chart(chart_df[["vibration"]], height=280)

    col_pressure, col_rpm = st.columns(2)
    with col_pressure:
        st.markdown("**Pressure**")
        st.line_chart(chart_df[["pressure"]], height=250)
    with col_rpm:
        st.markdown("**Rotational Speed**")
        st.line_chart(chart_df[["rotational_speed"]], height=250)

    st.markdown("**Predicted RUL Trend (last 50 readings)**")
    st.line_chart(chart_df[["predicted_rul_hours"]], height=250)

    # ---------------- Raw data table ----------------
    with st.expander("🔍 View Raw Telemetry Table"):
        display_cols = [
            "timestamp", "machine_id", "cycle", "temperature", "vibration",
            "pressure", "rotational_speed", "predicted_rul_hours",
        ]
        st.dataframe(
            featured_df[display_cols].sort_values("timestamp", ascending=False),
            use_container_width=True,
            height=300,
        )

    st.caption(f"Auto-refreshing every {REFRESH_SECONDS}s • Last render: {datetime.now().strftime('%H:%M:%S')}")


# --------------------------------------------------------------------------
# Page layout
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
def main():
    st.title("🛠️ Enterprise Predictive Maintenance Dashboard")
    st.markdown(
        "Real-time Remaining Useful Life (RUL) estimation for turbofan-style machinery, "
        "modeled on the statistical structure of the NASA C-MAPSS dataset. "
        "Telemetry flows: **MQTT Publisher → MQTT Broker → Subscriber → SQLite → Dashboard**."
    )

    with st.sidebar:
        st.header("⚙️ System Status")
        model, scaler, feature_columns = load_artifacts()
        st.write("Model artifacts:", "✅ Loaded" if model is not None else "❌ Missing")
        try:
            total_rows = db.row_count()
            st.write("Telemetry rows in DB:", total_rows)
        except Exception:
            st.write("Telemetry rows in DB:", "DB not initialized")
        
        st.markdown("---")
        
        # 👇 UPDATE: Yahan humne query hata kar direct teeno machines fix kar di hain
        st.subheader("🏭 Select Industrial Asset")
        available_machines = ["machine_1", "machine_2", "machine_3"]
        selected_machine = st.selectbox("Choose Active Machine:", available_machines)
        
        st.markdown("---")
        st.subheader("Alert Thresholds")
        st.write(f"🔴 Critical: RUL < {CRITICAL_RUL_THRESHOLD} hrs")
        st.write(f"🟠 Warning: RUL < {WARNING_RUL_THRESHOLD} hrs")
        st.write(f"🟢 Nominal: RUL ≥ {WARNING_RUL_THRESHOLD} hrs")
        st.markdown("---")
        st.caption(
            "This dashboard auto-refreshes on its own. "
            "Ensure `mqtt_publisher.py` and `mqtt_subscriber.py` are running "
            "in separate terminals for live data to appear."
        )

    # Selected machine ko hum dashboard renderer ke andar pass kar rahe hain
    render_live_dashboard(selected_machine)


if __name__ == "__main__":
    main()