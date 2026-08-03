"""
mqtt_subscriber.py
===================
Background ingestion service.

Subscribes to the live sensor topic ("factory/+/sensors"),
parses each incoming JSON payload, and continuously inserts it into
the local SQLite telemetry store (telemetry.db) via database_manager.

Run this alongside mqtt_publisher.py (which must be started first, or
started after -- order does not matter since MQTT will simply deliver
whatever is published while this subscriber is connected):

    python mqtt_subscriber.py
"""

import json
import argparse

import paho.mqtt.client as mqtt

import database_manager as db

BROKER_HOST = "localhost"
BROKER_PORT = 1883
TOPIC = "factory/+/sensors"

REQUIRED_FIELDS = [
    "machine_id", "cycle", "temperature", "vibration", "pressure",
    "rotational_speed", "operational_setting_1", "operational_setting_2",
    "operational_setting_3",
]


def on_connect(client, userdata, flags, reason_code, properties=None):
    if reason_code == 0 or reason_code == mqtt.MQTT_ERR_SUCCESS:
        print(f"[mqtt_subscriber] Connected to broker. Subscribing to '{userdata['topic']}'...")
        client.subscribe("factory/+/sensors")
    else:
        print(f"[mqtt_subscriber] Connection failed with reason code: {reason_code}")


def on_disconnect(client, userdata, reason_code, properties=None):
    print(f"[mqtt_subscriber] Disconnected from broker (reason: {reason_code}). Will attempt to reconnect.")


def on_subscribe(client, userdata, mid, reason_code_list=None, properties=None):
    print(f"[mqtt_subscriber] Subscription confirmed (mid={mid}).")


def on_message(client, userdata, msg):
    try:
        payload = json.loads(msg.payload.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        print(f"[mqtt_subscriber] WARNING: could not decode message on '{msg.topic}': {exc}")
        return

    missing = [f for f in REQUIRED_FIELDS if f not in payload]
    if missing:
        print(f"[mqtt_subscriber] WARNING: message missing fields {missing}, skipping.")
        return

    try:
        db.insert_telemetry(payload)
        print(f"[mqtt_subscriber] Inserted cycle {payload['cycle']} for {payload['machine_id']} "
              f"(temp={payload['temperature']}, vib={payload['vibration']})")
    except Exception as exc:
        print(f"[mqtt_subscriber] ERROR inserting telemetry into DB: {exc}")


def build_client(topic: str) -> mqtt.Client:
    userdata = {"topic": topic}
    try:
        client = mqtt.Client(
            mqtt.CallbackAPIVersion.VERSION2,
            client_id="telemetry_subscriber",
            userdata=userdata,
        )
    except AttributeError:
        client = mqtt.Client(client_id="telemetry_subscriber", userdata=userdata)

    client.on_connect = on_connect
    client.on_disconnect = on_disconnect
    client.on_subscribe = on_subscribe
    client.on_message = on_message
    client.reconnect_delay_set(min_delay=1, max_delay=30)
    return client


def main():
    parser = argparse.ArgumentParser(description="Subscribe to MQTT telemetry and persist to SQLite.")
    parser.add_argument("--host", default=BROKER_HOST, help="MQTT broker host")
    parser.add_argument("--port", type=int, default=BROKER_PORT, help="MQTT broker port")
    parser.add_argument("--topic", default=TOPIC, help="MQTT topic to subscribe to")
    args = parser.parse_args()

    print("[mqtt_subscriber] Initializing database...")
    db.init_db()

    client = build_client(args.topic)

    print(f"[mqtt_subscriber] Connecting to {args.host}:{args.port} ...")
    try:
        client.connect(args.host, args.port, keepalive=60)
    except Exception as exc:
        print(f"[mqtt_subscriber] ERROR: could not connect to broker: {exc}")
        print("[mqtt_subscriber] Make sure Mosquitto (or another MQTT broker) is running.")
        return

    print("[mqtt_subscriber] Listening for telemetry. Press Ctrl+C to stop.")
    try:
        client.loop_forever(retry_first_connection=True)
    except KeyboardInterrupt:
        print("\n[mqtt_subscriber] Stopping subscriber...")
        client.disconnect()


if __name__ == "__main__":
    main()
