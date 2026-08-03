# 🚀 Real-Time Predictive Maintenance & Remaining Useful Life (RUL) Estimation System

## 📌 Overview

This project is a **Real-Time Predictive Maintenance & Remaining Useful Life (RUL) Estimation System** designed to monitor industrial machinery using live sensor data and estimate how many operational hours remain before a machine is likely to fail.

Instead of waiting for equipment to break down, the system continuously analyzes sensor readings and predicts the machine's health, enabling maintenance teams to take proactive action and reduce unexpected downtime.

---

## 🎯 Problem Statement

Traditional maintenance strategies have several limitations:

### 🔴 Reactive Maintenance

* Maintenance is performed only after equipment fails.
* Results in unexpected downtime, production loss, and higher repair costs.

### 🟡 Preventive Maintenance

* Maintenance is scheduled at fixed intervals regardless of machine condition.
* Can lead to unnecessary maintenance and increased operational expenses.

### 🟢 Proposed Solution

This project implements **Predictive Maintenance**, where machine health is continuously monitored and the **Remaining Useful Life (RUL)** is estimated using Machine Learning. This helps organizations schedule maintenance only when required, improving reliability while reducing costs.

---

## 🛠️ Technology Stack

| Technology                   | Purpose                             |
| ---------------------------- | ----------------------------------- |
| **Python**                   | Core programming language           |
| **MQTT (Eclipse Mosquitto)** | Real-time sensor data communication |
| **SQLite**                   | Lightweight telemetry database      |
| **XGBoost Regressor**        | Remaining Useful Life prediction    |
| **Streamlit**                | Interactive real-time dashboard     |

---

## 🏗️ System Architecture

```text
Simulated Sensors
        │
        ▼
 mqtt_publisher.py
        │
        ▼
 MQTT Broker (Mosquitto)
        │
        ▼
 mqtt_subscriber.py
        │
        ▼
 SQLite Database (telemetry.db)
        │
        ▼
 Streamlit Dashboard (app.py)
        │
        ▼
 ML Model (XGBoost)
        │
        ▼
 Remaining Useful Life Prediction
```

---

## ⚙️ How It Works

### 1️⃣ Real-Time Sensor Data Generation

The `mqtt_publisher.py` script simulates industrial sensor readings including:

* Temperature
* Vibration
* Pressure
* Rotational Speed (RPM)

These readings are continuously published to the MQTT broker.

---

### 2️⃣ Data Collection & Storage

The `mqtt_subscriber.py` script listens to the MQTT topics, receives the incoming sensor data, and stores it in the SQLite database (`telemetry.db`) for further analysis.

---

### 3️⃣ Remaining Useful Life Prediction

The Streamlit dashboard continuously reads the latest telemetry data, generates rolling-window features, and feeds them into a trained **XGBoost Regressor** model.

The model predicts the **Remaining Useful Life (RUL)** of the machine in real time.

---

## 📊 Dashboard Features

The dashboard provides:

* 📈 Live sensor visualization
* 📉 Historical telemetry trends
* 🤖 Real-time RUL prediction
* 📊 Machine health monitoring
* ⚠️ Intelligent maintenance alerts

---

## 🚨 Alert Levels

### 🟢 Nominal (Safe)

* Remaining Useful Life > **40 Hours**
* Machine is operating normally.
* No maintenance required.

---

### 🟠 Caution (Warning)

* Remaining Useful Life between **15–40 Hours**
* Maintenance planning is recommended.

---

### 🔴 Critical (Danger)

* Remaining Useful Life < **15 Hours**
* Immediate maintenance is required.
* Dashboard displays a critical alert for rapid response.

---

## 📂 Project Structure

```text
Predictive-Maintenance/
│
├── app.py
├── mqtt_publisher.py
├── mqtt_subscriber.py
├── database_manager.py
├── telemetry.db
├── rul_model.pkl
├── scaler.pkl
├── feature_columns.pkl
├── requirements.txt
├── README.md
└── images/
```

---

## ▶️ Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/your-repository.git
cd your-repository
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Start MQTT Broker

Run the Eclipse Mosquitto broker.

---

### 4. Start Sensor Simulation

```bash
python mqtt_publisher.py
```

---

### 5. Start Data Subscriber

```bash
python mqtt_subscriber.py
```

---

### 6. Launch Dashboard

```bash
streamlit run app.py
```

---

## 💡 Applications

This system can be adapted for predictive maintenance in:

* Industrial Manufacturing
* Aerospace Engines
* Military Equipment
* Heavy Vehicles
* Railway Systems
* Power Plants
* Smart Factories
* Oil & Gas Equipment

---

## 📈 Future Improvements

* Cloud deployment (AWS / Azure)
* Real IoT sensor integration
* Email & SMS alert system
* Mobile application
* Multi-machine monitoring
* Advanced anomaly detection
* Docker & Kubernetes deployment

---

## 👨‍💻 Author

**Priyanshu Thakur**

B.Tech (Computer Science & Engineering)

Project: **Real-Time Predictive Maintenance & Remaining Useful Life (RUL) Estimation System**

---

## ⭐ If you found this project useful, consider giving it a Star on GitHub.
