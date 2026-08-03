
@echo off
TITLE Launching Predictive Maintenance System
COLOR 0A

echo Starting Telemetry Subscriber...
start "MQTT Subscriber" cmd /k "python mqtt_subscriber.py"

echo Starting Live Edge Simulator (Publisher)...
start "MQTT Publisher" cmd /k "python mqtt_publisher.py"

echo Launching Streamlit Dashboard...
start "Streamlit App" cmd /k "python -m streamlit run app.py"

:: Wait 6 seconds for Streamlit server to load properly
echo Waiting for Streamlit server to load...
timeout /t 6 /nobreak >nul

echo Opening Browser...
start http://localhost:8501

echo All services initiated successfully!