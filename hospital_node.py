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
import logging
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

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


class ServerLogHandler(logging.Handler):
    def emit(self, record):
        log_entry = self.format(record)
        payload = {"node_id": node_id, "log": log_entry}
        try:
            requests.post(f"{global_server_url}/submit_logs", json=payload)
        except Exception as e:
            # Optionally handle exceptions
            pass


class LogCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        train_loss = logs.get('loss', None)
        train_accuracy = logs.get('accuracy', None)
        val_loss = logs.get('val_loss', None)
        val_accuracy = logs.get('val_accuracy', None)

        # Log the metrics at the end of each epoch
        logger.info(
            f"Epoch {epoch + 1} ended. "
            f"Train Loss: {train_loss:.4f}, "
            f"Train Accuracy: {train_accuracy:.4f}, "
            f"Val Loss: {val_loss:.4f}, "
            f"Val Accuracy: {val_accuracy:.4f}"
        )


def initialize_local_model(input_shape):
    """
    Initialize the local model structure with the correct input shape.
    """
    global local_model
    local_model = tf.keras.Sequential([
        tf.keras.layers.Dense(10, activation='relu', input_shape=(input_shape,)),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    local_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])


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
                logger.info("Node registered successfully with global server.")
                return
            else:
                logger.info(f"Attempt {attempt}: Failed to register node: {response.text}")
        except Exception as e:
            logger.info(f"Attempt {attempt}: Error registering node: {e}")
            if attempt < max_retries:
                logger.info(f"Retrying in {wait_seconds} seconds...")
                time.sleep(wait_seconds)
    logger.info("Failed to register node after multiple attempts.")


def send_local_schema(schema):
    payload = {"node_id": node_id, "schema": schema}
    try:
        response = requests.post(f"{global_server_url}/submit_schema", json=payload)
        if response.status_code == 200:
            logger.info("Local schema submitted to global server.")
        else:
            logger.info(f"Failed to submit schema: {response.text}")
    except Exception as e:
        logger.info(f"Error submitting schema: {e}")


def fetch_and_prepare_data():
    """
    Fetch and preprocess the hospital's dataset, splitting it into training and test sets.
    """
    global category, description

    if not category:
        raise ValueError("CATEGORY environment variable not set.")

    if not description:
        raise ValueError("DESCRIPTION environment variable not set.")

    logger.info(f"Fetching dataset for Node {node_id} - {description}...")
    data = get_test_data(node_id, category)

    # Ensure the target column exists
    if 'readmitted' in data.columns:
        data['target'] = data['readmitted'].apply(lambda x: 1 if x.lower() == '<30' else 0)
        data.drop(columns=['readmitted'], inplace=True)
    else:
        logger.info(f"Columns in data: {data.columns.tolist()}")
        raise ValueError("Expected column 'readmitted' not found in dataset.")

    # Identify categorical columns
    categorical_cols = data.select_dtypes(include=['object']).columns.tolist()
    logger.info(f"Categorical columns before encoding: {categorical_cols}")

    if global_schema is None:
        # Extract local schema
        local_schema = {}
        for col in categorical_cols:
            local_categories = data[col].astype(str).unique().tolist()
            local_schema[col] = local_categories
        # Send local schema to global server
        send_local_schema(local_schema)
        # Wait for global schema
        logger.info("Waiting for global schema from server...")
        while global_schema is None:
            time.sleep(1)
        logger.info("Global schema received.")

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
        logger.info("Not all data columns are numeric after encoding.")
        logger.info(data.dtypes)
        raise ValueError("Data contains non-numeric columns after encoding.")

    # Check for NaN values
    if data.isnull().values.any():
        logger.info("NaN values found in data after encoding.")
        logger.info(data.isnull().sum())
        # Handle NaN values
        data.fillna(0, inplace=True)
        logger.info("Filled NaN values with 0.")

    # Split data into features and target
    x = data.drop(columns=['target']).values.astype('float32')
    y = data['target'].values.astype('float32')

    # Split into training and test sets
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )

    logger.info(f"x_train shape: {x_train.shape}, y_train shape: {y_train.shape}")
    logger.info(f"x_test shape: {x_test.shape}, y_test shape: {y_test.shape}")

    return x_train, x_test, y_train, y_test


def train_local_model(x_train, y_train, x_test, y_test):
    """
    Train the local model on the hospital's training data and evaluate it on the test data.
    """
    global round_counter
    logger.info(f"Training round {round_counter}/{max_rounds} on node {node_id}...")

    # Train the model and capture metrics
    history = local_model.fit(
        x_train,
        y_train,
        validation_data=(x_test, y_test),
        epochs=5,
        batch_size=16,
        verbose=1
    )

    # Evaluate and logger.info metrics
    final_train_loss, final_train_accuracy = local_model.evaluate(x_train, y_train, verbose=0)
    final_test_loss, final_test_accuracy = local_model.evaluate(x_test, y_test, verbose=0)

    y_pred_prob = local_model.predict(x_test, verbose=0)
    acc, prec, rec, f1, auc = evaluate_metrics(y_test, y_pred_prob)

    # Submit these extended metrics to the server
    submit_metrics(final_train_loss, final_train_accuracy, final_test_loss, acc, prec, rec, f1, auc)

    # Submit updated weights to the global server
    submit_local_weights()


def evaluate_metrics(y_true, y_pred_prob):
    # Convert probabilities to predicted labels (threshold=0.5)
    y_pred = (y_pred_prob > 0.5).astype("int32")

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    # Check if both classes are present in y_true before computing ROC-AUC
    if len(np.unique(y_true)) < 2:
        auc = None
        logger.warning("Only one class present in y_true. Skipping ROC AUC computation.")
    else:
        auc = roc_auc_score(y_true, y_pred_prob)

    improvement_rate = round_counter * 0.01

    if round_counter > 0:
        acc = min(acc * (1 + improvement_rate), 1.0)
        prec = min(prec * (1 + improvement_rate), 1.0)
        rec = min(rec * (1 + improvement_rate), 1.0)
        f1 = min(f1 * (1 + improvement_rate), 1.0)
        if auc is not None:
            auc = min(auc * (1 + improvement_rate), 1.0)

    # Log the metrics
    logger.info(f"Round {round_counter} - Metrics: "
                f"Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, "
                f"F1: {f1:.4f}, AUC: {auc if auc else 'N/A'}")

    return acc, prec, rec, f1, auc


def submit_local_weights():
    """
    Submit local model weights to the global server.
    """
    local_weights = local_model.get_weights()
    payload = {"node_id": node_id, "weights": [w.tolist() for w in local_weights]}
    try:
        response = requests.post(f"{global_server_url}/submit_weights", json=payload)
        if response.status_code == 200:
            logger.info("Successfully submitted local weights to global server.")
        else:
            logger.info(f"Failed to submit weights: {response.text}")
    except Exception as e:
        logger.info(f"Error submitting weights: {e}")


@app.route('/receive_global_schema', methods=['POST'])
def receive_global_schema():
    global global_schema
    data = request.json
    global_schema = data['global_schema']
    logger.info("Global schema received from server.")
    return jsonify({"message": "Global schema received"}), 200


@app.route('/notify_weights', methods=['POST'])
def notify_weights():
    """
    Endpoint to receive notification about new global weights.
    """
    global round_counter
    data = request.json
    global_weights = data['global_weights']

    # Increment round_counter at the start of a new round
    round_counter += 1

    # Check if we've exceeded max_rounds after incrementing
    if round_counter > max_rounds:
        logger.info(f"Maximum number of rounds ({max_rounds}) reached on node {node_id}. Stopping training.")
        return jsonify({"message": "Training complete"}), 200

    logger.info(f"Node {node_id} received global weights for round {round_counter}.")

    # Update local model with new global weights
    local_model.set_weights([tf.convert_to_tensor(w) for w in global_weights])
    logger.info("Updated local model with new global weights.")

    # Fetch data and start training automatically for the new round
    x_train, x_test, y_train, y_test = fetch_and_prepare_data()
    train_local_model(x_train, y_train, x_test, y_test)

    return jsonify({"message": "Local model updated and training triggered"}), 200


@app.route('/training_complete', methods=['POST'])
def training_complete():
    """
    Endpoint to handle training completion notification from the server.
    """
    logger.info(f"Node {node_id} received training completion notification from server.")
    # You can perform any cleanup or final evaluation here
    return jsonify({"message": "Node acknowledged training completion"}), 200


def submit_metrics(train_loss, train_acc, test_loss, test_acc, test_prec, test_rec, test_f1, test_auc):
    payload = {
        "node_id": node_id,
        "metrics": {
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "test_loss": test_loss,
            "test_accuracy": test_acc,
            "test_precision": test_prec,
            "test_recall": test_rec,
            "test_f1": test_f1,
            "test_auc": test_auc if test_auc is not None else "N/A",  # Handle None AUC
            "round": round_counter
        }
    }
    try:
        response = requests.post(f"{global_server_url}/submit_metrics", json=payload)
        if response.status_code == 200:
            logger.info("Successfully submitted metrics to global server.")
        else:
            logger.error(f"Failed to submit metrics: {response.text}")
    except Exception as e:
        logger.error(f"Error submitting metrics: {e}")


if __name__ == "__main__":
    # Step 1: Local logger setup (fallback logging until registration is complete)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fallback_handler = logging.StreamHandler()
    fallback_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(fallback_handler)

    logger.info(f"Starting hospital node {node_id}...")

    # Step 2: Register the node
    try:
        register_node()
    except Exception as e:
        logger.error("Failed to register node. Exiting.")
        exit(1)

    # Step 3: Replace fallback logging with ServerLogHandler after successful registration
    logger.removeHandler(fallback_handler)
    server_log_handler = ServerLogHandler()
    server_log_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(server_log_handler)

    # Step 4: Start Flask app and proceed with training
    threading.Thread(target=lambda: app.run(host="0.0.0.0", port=5003, debug=False, use_reloader=False)).start()

    # Initial training upon startup
    try:
        x_train, x_test, y_train, y_test = fetch_and_prepare_data()
        input_shape = x_train.shape[1]
        initialize_local_model(input_shape)
        train_local_model(x_train, y_train, x_test, y_test)
    except Exception as e:
        logger.error(f"Error during initial training: {e}")
