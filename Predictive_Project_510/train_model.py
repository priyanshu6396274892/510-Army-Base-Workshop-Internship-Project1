"""
train_model.py
================
Principal IoT Architect & Senior Lead Data Scientist deliverable.

Simulates the statistical structure of the NASA C-MAPSS turbofan engine
degradation dataset (multiple units, multiple operating conditions,
sensor drift toward failure, piecewise-linear Remaining Useful Life
target) and trains a gradient-boosted regression model (XGBoost) to
predict RUL from live sensor telemetry.

Outputs
-------
rul_model.pkl   - trained XGBoost Regressor
scaler.pkl      - fitted StandardScaler for the feature vector
feature_columns.pkl - ordered list of feature names the model expects
                       (kept in sync with app.py)

Run:
    python train_model.py
"""

import numpy as np
import pandas as pd
import joblib
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
RANDOM_SEED = 42
N_UNITS = 120                # number of simulated "engines" (C-MAPSS style units)
MIN_LIFE_CYCLES = 150
MAX_LIFE_CYCLES = 350
ROLLING_WINDOW = 5
RUL_CLIP = 125                # piecewise-linear RUL cap, standard C-MAPSS practice
MODEL_PATH = "rul_model.pkl"
SCALER_PATH = "scaler.pkl"
FEATURE_COLUMNS_PATH = "feature_columns.pkl"

np.random.seed(RANDOM_SEED)

# Base raw sensor / operating-setting columns that arrive over MQTT
BASE_SENSOR_COLUMNS = [
    "temperature",
    "vibration",
    "pressure",
    "rotational_speed",
    "operational_setting_1",
    "operational_setting_2",
    "operational_setting_3",
]

# Sensors we compute rolling statistics on (degradation-sensitive channels)
ROLLING_SENSORS = ["temperature", "vibration", "pressure", "rotational_speed"]

# Final ordered feature vector used by the model (must match app.py exactly)
FEATURE_COLUMNS = BASE_SENSOR_COLUMNS + [
    f"{s}_rolling_mean_{ROLLING_WINDOW}" for s in ROLLING_SENSORS
] + [
    f"{s}_rolling_std_{ROLLING_WINDOW}" for s in ROLLING_SENSORS
]


# --------------------------------------------------------------------------
# Synthetic C-MAPSS-style trajectory generator
# --------------------------------------------------------------------------
def _generate_unit_trajectory(unit_id: int) -> pd.DataFrame:
    """
    Generates one simulated engine's full run-to-failure trajectory.

    Mirrors C-MAPSS behavior:
      - Sensors are flat/noisy while the unit is healthy.
      - Beyond ~60% of life, degradation accelerates non-linearly.
      - Operating conditions vary cycle to cycle (3 operational settings).
    """
    max_cycles = np.random.randint(MIN_LIFE_CYCLES, MAX_LIFE_CYCLES)
    cycles = np.arange(1, max_cycles + 1)
    life_fraction = cycles / max_cycles

    # Degradation severity ramps up non-linearly as the unit approaches failure
    degradation = np.where(
        life_fraction < 0.6,
        life_fraction * 0.15,
        0.09 + (np.maximum(life_fraction - 0.6, 0.0) ** 2.2) * 3.5,
    )

    # Operating settings: slow random walk within realistic bounds
    op1 = np.clip(np.cumsum(np.random.normal(0, 0.02, max_cycles)) + 20, 0, 42)
    op2 = np.clip(np.cumsum(np.random.normal(0, 0.005, max_cycles)) + 0.7, 0, 1)
    op3 = np.full(max_cycles, 100.0) + np.random.normal(0, 0.5, max_cycles)

    # Sensor channels: physically-plausible drift directions
    temperature = (
        550 + degradation * 220 + np.random.normal(0, 2.0, max_cycles) + (op1 - 20) * 0.3
    )
    vibration = (
        0.20 + degradation * 3.6 + np.random.normal(0, 0.03, max_cycles)
    )
    pressure = (
        150 - degradation * 55 + np.random.normal(0, 1.2, max_cycles) - (op2 - 0.7) * 4
    )
    rotational_speed = (
        3600 - degradation * 900 + np.random.normal(0, 8.0, max_cycles)
    )

    # Piecewise-linear RUL target (standard C-MAPSS convention: cap early life)
    raw_rul = max_cycles - cycles
    rul = np.clip(raw_rul, 0, RUL_CLIP)

    df = pd.DataFrame({
        "unit_id": unit_id,
        "cycle": cycles,
        "temperature": temperature,
        "vibration": vibration,
        "pressure": pressure,
        "rotational_speed": rotational_speed,
        "operational_setting_1": op1,
        "operational_setting_2": op2,
        "operational_setting_3": op3,
        "RUL": rul,
    })
    return df


def generate_cmapss_style_dataset(n_units: int = N_UNITS) -> pd.DataFrame:
    """Builds the full multi-unit synthetic training dataset."""
    frames = [_generate_unit_trajectory(uid) for uid in range(1, n_units + 1)]
    data = pd.concat(frames, ignore_index=True)
    return data


def engineer_rolling_features(df: pd.DataFrame, window: int = ROLLING_WINDOW) -> pd.DataFrame:
    """
    Adds rolling mean / std features per unit, matching how the live
    dashboard computes rolling statistics over a sliding telemetry window.
    """
    df = df.sort_values(["unit_id", "cycle"]).copy()
    for sensor in ROLLING_SENSORS:
        grouped = df.groupby("unit_id")[sensor]
        df[f"{sensor}_rolling_mean_{window}"] = (
            grouped.transform(lambda s: s.rolling(window, min_periods=1).mean())
        )
        df[f"{sensor}_rolling_std_{window}"] = (
            grouped.transform(lambda s: s.rolling(window, min_periods=1).std().fillna(0.0))
        )
    return df


# --------------------------------------------------------------------------
# Training pipeline
# --------------------------------------------------------------------------
def train():
    print("[1/5] Simulating NASA C-MAPSS-style turbofan degradation dataset...")
    raw = generate_cmapss_style_dataset()
    print(f"      Generated {raw['unit_id'].nunique()} units, {len(raw)} total cycles.")

    print("[2/5] Engineering rolling-window degradation features...")
    featured = engineer_rolling_features(raw)

    X = featured[FEATURE_COLUMNS]
    y = featured["RUL"]

    print("[3/5] Splitting train/test and fitting StandardScaler...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("[4/5] Training XGBoost Regressor...")
    model = XGBRegressor(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=1.0,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        objective="reg:squarederror",
    )
    model.fit(X_train_scaled, y_train)

    preds = model.predict(X_test_scaled)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    print("[5/5] Evaluation on held-out test set:")
    print(f"      MAE  : {mae:.2f} cycles")
    print(f"      RMSE : {rmse:.2f} cycles")
    print(f"      R^2  : {r2:.4f}")

    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(FEATURE_COLUMNS, FEATURE_COLUMNS_PATH)

    print(f"\nSaved model -> {MODEL_PATH}")
    print(f"Saved scaler -> {SCALER_PATH}")
    print(f"Saved feature schema -> {FEATURE_COLUMNS_PATH}")
    print("\nTraining complete. You can now run mqtt_publisher.py, mqtt_subscriber.py, and app.py.")


if __name__ == "__main__":
    train()
