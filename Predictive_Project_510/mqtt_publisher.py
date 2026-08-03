"""
mqtt_publisher.py
==================
Live sensor simulator ("Edge Device" stand-in).

Simulates multiple turbofan-style machines running through independent degradation
trajectories and publishes JSON-encoded sensor readings to a local MQTT
broker (e.g. Mosquitto) on dynamic topics:

    factory/machine_1/sensors
    factory/machine_2/sensors
    factory/machine_3/sensors

Every message contains: timestamp, machine_id, cycle, temperature,
vibration, pressure, rotational_speed, and three operational settings.

Requires a running MQTT broker (see README instructions). Default
assumes Mosquitto running on localhost:1883 with no auth.

Run:
    python mqtt_publisher.py
"""

import json
import time
import argparse
import random
from datetime import datetime, timezone

import numpy as np
import paho.mqtt.client as mqtt

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
BROKER_HOST = "localhost"
BROKER_PORT = 1883
TOPIC = "factory/machine_1/sensors"
PUBLISH_INTERVAL_SECONDS = 2.0
TOTAL_LIFE_CYCLES = 250   # simulated total life of this run before "failure"/reset


class EngineSimulator:
    """
    Generates one cycle of sensor telemetry at a time, walking the machine
    from healthy operation toward failure, then looping back to a fresh
    simulated unit (mirrors a fleet where machines get serviced/replaced).
    """

    def __init__(self, machine_id: str, total_life_cycles: int = TOTAL_LIFE_CYCLES, seed: int = None):
        self.machine_id = machine_id  # Assigned machine ID dynamically
        self.total_life_cycles = total_life_cycles
        self.rng = np.random.default_rng(seed)
        self._reset()

    def _reset(self):
        self.cycle = 0
        self.total_life_cycles = random.randint(180, 300)
        print(f"[simulator] New unit online for {self.machine_id}. Simulated life span: {self.total_life_cycles} cycles.")

    def next_reading(self) -> dict:
        self.cycle += 1
        life_fraction = self.cycle / self.total_life_cycles

        degradation = (
            life_fraction * 0.15
            if life_fraction < 0.6
            else 0.09 + (max(life_fraction - 0.6, 0.0) ** 2.2) * 3.5
        )

        op1 = float(np.clip(20 + self.rng.normal(0, 0.6), 0, 42))
        op2 = float(np.clip(0.7 + self.rng.normal(0, 0.02), 0, 1))
        op3 = float(100 + self.rng.normal(0, 0.5))

        temperature = 550 + degradation * 220 + self.rng.normal(0, 2.0) + (op1 - 20) * 0.3
        vibration = 0.20 + degradation * 3.6 + self.rng.normal(0, 0.03)
        pressure = 150 - degradation * 55 + self.rng.normal(0, 1.2) - (op2 - 0.7) * 4
        rotational_speed = 3600 - degradation * 900 + self.rng.normal(0, 8.0)

        reading = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "machine_id": self.machine_id,  # Uses the instance's unique machine ID
            "cycle": self.cycle,
            "temperature": round(float(temperature), 3),
            "vibration": round(float(vibration), 4),
            "pressure": round(float(pressure), 3),
            "rotational_speed": round(float(rotational_speed), 2),
            "operational_setting_1": round(op1, 3),
            "operational_setting_2": round(op2, 4),
            "operational_setting_3": round(op3, 3),
        }

        # unit "fails" and gets replaced with a fresh one at end of life
        if self.cycle >= self.total_life_cycles:
            print(f"[simulator] {self.machine_id} reached end-of-life at cycle {self.cycle}. Cycling to next unit.")
            self._reset()

        return reading


def on_connect(client, userdata, flags, reason_code, properties=None):
    if reason_code == 0 or reason_code == mqtt.MQTT_ERR_SUCCESS:
        print(f"[mqtt_publisher] Connected to broker at {BROKER_HOST}:{BROKER_PORT}")
    else:
        print(f"[mqtt_publisher] Connection failed with reason code: {reason_code}")


def on_disconnect(client, userdata, reason_code, properties=None):
    print(f"[mqtt_publisher] Disconnected from broker (reason: {reason_code}). Will attempt to reconnect.")


def build_client() -> mqtt.Client:
    try:
        # paho-mqtt >= 2.0 API
        client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="engine_publisher")
    except AttributeError:
        # paho-mqtt 1.x fallback
        client = mqtt.Client(client_id="engine_publisher")
    client.on_connect = on_connect
    client.on_disconnect = on_disconnect
    client.reconnect_delay_set(min_delay=1, max_delay=30)
    return client


def main():
    parser = argparse.ArgumentParser(description="Publish simulated live engine telemetry to MQTT.")
    parser.add_argument("--host", default=BROKER_HOST, help="MQTT broker host")
    parser.add_argument("--port", type=int, default=BROKER_PORT, help="MQTT broker port")
    parser.add_argument("--topic", default=TOPIC, help="MQTT topic template to publish to")
    parser.add_argument("--interval", type=float, default=PUBLISH_INTERVAL_SECONDS,
                        help="Seconds between published readings")
    args = parser.parse_args()

    client = build_client()

    print(f"[mqtt_publisher] Connecting to {args.host}:{args.port} ...")
    try:
        client.connect(args.host, args.port, keepalive=60)
    except Exception as exc:
        print(f"[mqtt_publisher] ERROR: could not connect to broker: {exc}")
        print("[mqtt_publisher] Make sure Mosquitto (or another MQTT broker) is running.")
        return

    client.loop_start()

    # Create 3 separate machine simulators with distinct random seeds
    machines_list = ["machine_1", "machine_2", "machine_3"]
    simulators = [EngineSimulator(machine_id=m, seed=i*10) for i, m in enumerate(machines_list)]

    print(f"[mqtt_publisher] Streaming simulated telemetry for {machines_list} every {args.interval}s.")
    print("Press Ctrl+C to stop.\n")

    try:
        while True:
            for simulator in simulators:
                reading = simulator.next_reading()
                payload = json.dumps(reading)
                
                # Dynamically change the topic name based on current machine id
                dynamic_topic = args.topic.replace("machine_1", simulator.machine_id)
                
                result = client.publish(dynamic_topic, payload, qos=1)
                status = result[0]
                if status == mqtt.MQTT_ERR_SUCCESS:
                    print(f"[mqtt_publisher] Published {simulator.machine_id:9s} | cycle {reading['cycle']:4d} | "
                          f"Temp={reading['temperature']:.1f} | Vib={reading['vibration']:.3f} | "
                          f"Pressure={reading['pressure']:.1f} | RPM={reading['rotational_speed']:.0f}")
                else:
                    print(f"[mqtt_publisher] WARNING: publish failed for {simulator.machine_id} with status {status}")
            
            print("-" * 80) # Visual separator for each timeframe batch
            time.sleep(args.interval)
            
    except KeyboardInterrupt:
        print("\n[mqtt_publisher] Stopping publisher...")
    finally:
        client.loop_stop()
        client.disconnect()


if __name__ == "__main__":
    main()