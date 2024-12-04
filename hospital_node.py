from flask import Flask, request, jsonify
import tensorflow as tf
import requests
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from data_helper import get_test_data
import time

app = Flask(__name__)

# Global variables
local_model = None
global_server_url = os.environ.get("GLOBAL_SERVER_URL")
node_id = os.environ.get("NODE_ID")
webhook_url = os.environ.get("WEBHOOK_URL")
category = os.environ.get("CATEGORY")
description = os.environ.get("DESCRIPTION")


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

    # Encode categorical variables
    if categorical_cols:
        data = pd.get_dummies(data, columns=categorical_cols, drop_first=True)

    # Verify no object columns remain
    non_numeric_cols = data.select_dtypes(include=['object']).columns.tolist()
    if non_numeric_cols:
        print(f"Non-numeric columns after encoding: {non_numeric_cols}")
        raise ValueError("There are still non-numeric columns after encoding.")

    # Ensure all data is numeric
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
    print("Training local model...")

    # Train the model and capture metrics
    history = local_model.fit(
        x_train,
        y_train,
        validation_data=(x_test, y_test),
        epochs=5,
        batch_size=32,
        verbose=1
    )

    # Print final training metrics
    final_train_loss, final_train_accuracy = local_model.evaluate(x_train, y_train, verbose=0)
    print(f"Final Training Metrics: Loss = {final_train_loss:.4f}, Accuracy = {final_train_accuracy:.4f}")

    # Print final test metrics
    final_test_loss, final_test_accuracy = local_model.evaluate(x_test, y_test, verbose=0)
    print(f"Final Test Metrics: Loss = {final_test_loss:.4f}, Accuracy = {final_test_accuracy:.4f}")

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


@app.route('/notify_weights', methods=['POST'])
def notify_weights():
    """
    Endpoint to receive notification about new global weights.
    """
    data = request.json
    global_weights = data['global_weights']

    # Update local model with new global weights
    local_model.set_weights([tf.convert_to_tensor(w) for w in global_weights])
    print("Updated local model with new global weights.")

    # Fetch data and start training automatically
    x_train, x_test, y_train, y_test = fetch_and_prepare_data()
    train_local_model(x_train, y_train, x_test, y_test)

    return jsonify({"message": "Local model updated and training triggered"}), 200


if __name__ == "__main__":
    print(f"Starting hospital node {node_id}...")
    register_node()

    # Initial training upon startup
    try:
        x_train, x_test, y_train, y_test = fetch_and_prepare_data()
        input_shape = x_train.shape[1]
        initialize_local_model(input_shape)
        train_local_model(x_train, y_train, x_test, y_test)
    except Exception as e:
        print(f"Error during initial training: {e}")

    app.run(host="0.0.0.0", port=5003, debug=False)
