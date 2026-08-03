# Enterprise Predictive Maintenance System (C-MAPSS-style RUL Prediction)

Pipeline: **MQTT Publisher (edge sim) → Mosquitto Broker → MQTT Subscriber → SQLite → Streamlit Dashboard**

## Files

| File | Purpose |
|---|---|
| `requirements.txt` | Python dependencies |
| `train_model.py` | Simulates C-MAPSS-style degradation data, trains XGBoost RUL model |
| `mqtt_publisher.py` | Simulates a live degrading machine, publishes sensor JSON over MQTT |
| `database_manager.py` | SQLite schema + read/write data-access layer (`telemetry.db`) |
| `mqtt_subscriber.py` | Subscribes to MQTT topic, writes readings into SQLite |
| `app.py` | Streamlit dashboard: live RUL inference, gauge, trend charts, alerts |

---

## Step 1 — Install a local MQTT broker (Mosquitto)

You need a running MQTT broker on `localhost:1883` before starting the publisher/subscriber.

### macOS (Homebrew)
```bash
brew install mosquitto
brew services start mosquitto
```

### Windows
1. Download the installer from https://mosquitto.org/download/
2. Run the installer (default settings are fine).
3. Start the service:
```powershell
net start mosquitto
```
(If it's not registered as a service, run `"C:\Program Files\mosquitto\mosquitto.exe" -v` in a terminal instead.)

### Ubuntu / Debian Linux
```bash
sudo apt update
sudo apt install -y mosquitto mosquitto-clients
sudo systemctl enable mosquitto
sudo systemctl start mosquitto
```

### Verify the broker is running
```bash
mosquitto_sub -t "test/topic" &
mosquitto_pub -t "test/topic" -m "hello broker"
```
You should see `hello broker` printed. If so, Mosquitto is listening on the default port `1883`.

> Alternative: run Mosquitto in Docker if you don't want a local install:
> ```bash
> docker run -it -p 1883:1883 eclipse-mosquitto
> ```

---

## Step 2 — Set up the Python environment

```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

---

## Step 3 — Train the RUL model

```bash
python train_model.py
```
This generates `rul_model.pkl`, `scaler.pkl`, and `feature_columns.pkl` in the project directory. You'll see MAE/RMSE/R² metrics printed to confirm the model trained correctly.

---

## Step 4 — Run all components simultaneously

Open **four separate terminals** (all in the project directory, with the venv activated):

**Terminal 1 — MQTT broker** (skip if using a system service / Docker container already running):
```bash
mosquitto -v
```

**Terminal 2 — Database initialization + live subscriber** (creates `telemetry.db` automatically):
```bash
python mqtt_subscriber.py
```

**Terminal 3 — Live sensor publisher (simulated edge device):**
```bash
python mqtt_publisher.py
```

**Terminal 4 — Streamlit dashboard:**
```bash
streamlit run app.py
```

Then open the URL Streamlit prints (typically `http://localhost:8501`) in your browser.

---

## What you should see

- The dashboard auto-refreshes every 5 seconds.
- A **gauge meter** shows current machine health (0–125 hrs of RUL).
- A **countdown card** prominently displays "Predicted Remaining Useful Life: X Hours".
- **Live line charts** track temperature, vibration, pressure, and rotational speed.
- As the simulated machine approaches end-of-life, predicted RUL falls — once it drops below **15 hours**, a **flashing red critical alert banner** appears on the dashboard and a matching alert is logged to the `mqtt_subscriber.py`/Streamlit console.
- When the simulated unit "fails," `mqtt_publisher.py` automatically starts a fresh simulated unit so you can watch multiple degradation cycles without restarting anything.

---

## Notes on the synthetic data

Real NASA C-MAPSS data isn't public-domain to redistribute here, so `train_model.py` and `mqtt_publisher.py` generate synthetic run-to-failure trajectories that reproduce its key statistical properties: multiple operating conditions, gradual-then-accelerating sensor drift, rolling-window degradation features, and a piecewise-linear RUL target (capped at 125 cycles, the standard convention in C-MAPSS literature). Swap in real C-MAPSS `.txt` files and adjust the column mapping in `train_model.py` if you want to train on the authentic dataset instead.
