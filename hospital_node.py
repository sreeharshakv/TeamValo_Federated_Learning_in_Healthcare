from flask import Flask, request, jsonify
import tensorflow as tf
import requests
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from data_helper import get_test_data
import time
import threading

app = Flask(__name__)

# Global variables
local_model = None
global_server_url = os.environ.get("GLOBAL_SERVER_URL")
node_id = os.environ.get("NODE_ID")
webhook_url = os.environ.get("WEBHOOK_URL")
category = os.environ.get("CATEGORY")
description = os.environ.get("DESCRIPTION")
global_schema = None  # To be received from the server
round_counter = 0
max_rounds = 5  # Should match the server's max_rounds


def initialize_local_model(input_shape):
    """
    Initialize the local model structure with the correct input shape.
    """
    global local_model
    local_model = tf.keras.Sequential([
        tf.keras.layers.Dense(10, activation='relu', input_shape=(input_shape,)),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    local_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


def register_node():
    """
    Register the hospital node with the global server.
    """
    payload = {"node_id": node_id, "webhook_url": webhook_url}
    max_retries = 5
    wait_seconds = 5
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.post(f"{global_server_url}/register_node", json=payload)
            if response.status_code == 200:
                print("Node registered successfully with global server.")
                return
            else:
                print(f"Attempt {attempt}: Failed to register node: {response.text}")
        except Exception as e:
            print(f"Attempt {attempt}: Error registering node: {e}")
            if attempt < max_retries:
                print(f"Retrying in {wait_seconds} seconds...")
                time.sleep(wait_seconds)
    print("Failed to register node after multiple attempts.")


def send_local_schema(schema):
    payload = {"node_id": node_id, "schema": schema}
    try:
        response = requests.post(f"{global_server_url}/submit_schema", json=payload)
        if response.status_code == 200:
            print("Local schema submitted to global server.")
        else:
            print(f"Failed to submit schema: {response.text}")
    except Exception as e:
        print(f"Error submitting schema: {e}")


def fetch_and_prepare_data():
    """
    Fetch and preprocess the hospital's dataset, splitting it into training and test sets.
    """
    global category, description

    if not category:
        raise ValueError("CATEGORY environment variable not set.")

    if not description:
        raise ValueError("DESCRIPTION environment variable not set.")

    print(f"Fetching dataset for Node {node_id} - {description}...")
    data = get_test_data(node_id, category)

    # Ensure the target column exists
    if 'readmitted' in data.columns:
        data['target'] = data['readmitted'].apply(lambda x: 1 if x.lower() == '<30' else 0)
        data.drop(columns=['readmitted'], inplace=True)
    else:
        print(f"Columns in data: {data.columns.tolist()}")
        raise ValueError("Expected column 'readmitted' not found in dataset.")

    # Identify categorical columns
    categorical_cols = data.select_dtypes(include=['object']).columns.tolist()
    print(f"Categorical columns before encoding: {categorical_cols}")

    if global_schema is None:
        # Extract local schema
        local_schema = {}
        for col in categorical_cols:
            local_categories = data[col].astype(str).unique().tolist()
            local_schema[col] = local_categories
        # Send local schema to global server
        send_local_schema(local_schema)
        # Wait for global schema
        print("Waiting for global schema from server...")
        while global_schema is None:
            time.sleep(1)
        print("Global schema received.")

    # Encode categorical variables using global schema
    for col in categorical_cols:
        data[col] = data[col].astype(str)
        data[col] = pd.Categorical(data[col], categories=global_schema[col])
    data = pd.get_dummies(data, columns=categorical_cols, drop_first=False)

    # Reindex to ensure all expected columns are present
    expected_columns = []
    for col in global_schema.keys():
        categories = global_schema[col]
        expected_columns.extend([f"{col}_{category}" for category in categories])
    # Add other non-categorical columns
    non_categorical_cols = [col for col in data.columns if col not in expected_columns and col != 'target']
    expected_columns.extend(non_categorical_cols)

    # Reindex the DataFrame to ensure consistent column order and fill missing columns with zeros
    data = data.reindex(columns=expected_columns + ['target'], fill_value=0)

    # Verify that all data is numeric
    if not all([np.issubdtype(dtype, np.number) for dtype in data.dtypes]):
        print("Not all data columns are numeric after encoding.")
        print(data.dtypes)
        raise ValueError("Data contains non-numeric columns after encoding.")

    # Check for NaN values
    if data.isnull().values.any():
        print("NaN values found in data after encoding.")
        print(data.isnull().sum())
        # Handle NaN values
        data.fillna(0, inplace=True)
        print("Filled NaN values with 0.")

    # Split data into features and target
    x = data.drop(columns=['target']).values.astype('float32')
    y = data['target'].values.astype('float32')

    # Split into training and test sets
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )

    print(f"x_train shape: {x_train.shape}, y_train shape: {y_train.shape}")
    print(f"x_test shape: {x_test.shape}, y_test shape: {y_test.shape}")

    return x_train, x_test, y_train, y_test


def train_local_model(x_train, y_train, x_test, y_test):
    """
    Train the local model on the hospital's training data and evaluate it on the test data.
    """
    global round_counter
    round_counter += 1
    print(f"Training round {round_counter}/{max_rounds} on node {node_id}...")

    # Train the model and capture metrics
    history = local_model.fit(
        x_train,
        y_train,
        validation_data=(x_test, y_test),
        epochs=5,
        batch_size=32,
        verbose=1
    )

    # Evaluate and print metrics
    final_train_loss, final_train_accuracy = local_model.evaluate(x_train, y_train, verbose=0)
    final_test_loss, final_test_accuracy = local_model.evaluate(x_test, y_test, verbose=0)
    print(
        f"Round {round_counter} - Final Training Metrics: Loss = {final_train_loss:.4f}, Accuracy = {final_train_accuracy:.4f}")
    print(
        f"Round {round_counter} - Final Test Metrics: Loss = {final_test_loss:.4f}, Accuracy = {final_test_accuracy:.4f}")

    # Optionally log or save the metrics for later analysis

    # Submit updated weights to the global server
    submit_local_weights()


def submit_local_weights():
    """
    Submit local model weights to the global server.
    """
    local_weights = local_model.get_weights()
    payload = {"node_id": node_id, "weights": [w.tolist() for w in local_weights]}
    try:
        response = requests.post(f"{global_server_url}/submit_weights", json=payload)
        if response.status_code == 200:
            print("Successfully submitted local weights to global server.")
        else:
            print(f"Failed to submit weights: {response.text}")
    except Exception as e:
        print(f"Error submitting weights: {e}")


@app.route('/receive_global_schema', methods=['POST'])
def receive_global_schema():
    global global_schema
    data = request.json
    global_schema = data['global_schema']
    print("Global schema received from server.")
    return jsonify({"message": "Global schema received"}), 200


@app.route('/notify_weights', methods=['POST'])
def notify_weights():
    """
    Endpoint to receive notification about new global weights.
    """
    global round_counter
    data = request.json
    global_weights = data['global_weights']

    if round_counter >= max_rounds:
        print(f"Maximum number of rounds reached on node {node_id}. Stopping training.")
        return jsonify({"message": "Training complete"}), 200

    print(f"Node {node_id} received global weights for round {round_counter + 1}.")

    # Update local model with new global weights
    local_model.set_weights([tf.convert_to_tensor(w) for w in global_weights])
    print("Updated local model with new global weights.")

    # Fetch data and start training automatically
    x_train, x_test, y_train, y_test = fetch_and_prepare_data()
    train_local_model(x_train, y_train, x_test, y_test)

    return jsonify({"message": "Local model updated and training triggered"}), 200


@app.route('/training_complete', methods=['POST'])
def training_complete():
    """
    Endpoint to handle training completion notification from the server.
    """
    print(f"Node {node_id} received training completion notification from server.")
    # You can perform any cleanup or final evaluation here
    return jsonify({"message": "Node acknowledged training completion"}), 200


if __name__ == "__main__":
    print(f"Starting hospital node {node_id}...")
    register_node()

    # Start Flask app in a separate thread to handle incoming schema and weight updates
    threading.Thread(target=lambda: app.run(host="0.0.0.0", port=5003, debug=False, use_reloader=False)).start()

    # Initial training upon startup
    try:
        x_train, x_test, y_train, y_test = fetch_and_prepare_data()
        input_shape = x_train.shape[1]
        initialize_local_model(input_shape)
        train_local_model(x_train, y_train, x_test, y_test)
    except Exception as e:
        print(f"Error during initial training: {e}")
