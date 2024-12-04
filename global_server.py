from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import requests
from concurrent.futures import ThreadPoolExecutor

app = Flask(__name__)

# Global Model and Metadata
global_model = None
received_weights = {}
registered_nodes = {}


@app.route('/health', methods=['GET'])
def health():
    return 'OK', 200


def initialize_global_model():
    """
    Initialize the global model and return its initial weights.
    """
    global global_model
    global_model = tf.keras.Sequential([
        tf.keras.layers.Dense(10, activation='relu', input_shape=(10,)),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    global_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return global_model.get_weights()


def aggregate_weights(received_weights):
    """
    Perform Federated Averaging to aggregate weights.
    """
    print("Aggregating weights...")
    aggregated_weights = []
    for weights_list in zip(*received_weights):  # Iterate layer by layer
        aggregated_weights.append(np.mean(np.array(weights_list), axis=0))
    return aggregated_weights


@app.route('/register_node', methods=['POST'])
def register_node():
    """
    Register a hospital node with its webhook URL.
    """
    data = request.json
    node_id = data.get('node_id')
    webhook_url = data.get('webhook_url')

    if not node_id or not webhook_url:
        return jsonify({"message": "node_id and webhook_url are required"}), 400

    registered_nodes[node_id] = webhook_url
    print(f"Node {node_id} registered with webhook URL {webhook_url}")
    return jsonify({"message": f"Node {node_id} registered successfully"}), 200


@app.route('/submit_weights', methods=['POST'])
def submit_weights():
    """
    Endpoint for hospital nodes to submit updated weights.
    """
    global received_weights
    data = request.json
    node_id = data['node_id']
    weights = data['weights']

    print(f"Received weights from {node_id}.")
    received_weights[node_id] = np.array(weights)

    # Check if all registered nodes have submitted weights
    if len(received_weights) == len(registered_nodes):
        print("All nodes have submitted weights. Aggregating...")
        aggregated_weights = aggregate_weights(list(received_weights.values()))
        global_model.set_weights(aggregated_weights)
        received_weights = {}  # Clear buffer

        # Notify hospital nodes about new global weights
        notify_hospital_nodes()

    return jsonify({"message": "Weights received", "status": "success"}), 200


def notify_hospital_nodes():
    """
    Notify hospital nodes via their webhook endpoints using multithreading.
    """
    global global_model
    global_weights = global_model.get_weights()

    def notify_node(node_id, url):
        try:
            response = requests.post(url, json={"global_weights": [w.tolist() for w in global_weights]})
            if response.status_code == 200:
                print(f"Successfully notified {node_id}.")
            else:
                print(f"Failed to notify {node_id}: {response.status_code} {response.text}")
        except Exception as e:
            print(f"Error notifying {node_id}: {e}")

    with ThreadPoolExecutor() as executor:
        for node_id, webhook_url in registered_nodes.items():
            executor.submit(notify_node, node_id, webhook_url)


@app.route('/initialize_model', methods=['GET'])
def initialize_model():
    """
    Endpoint to initialize and distribute the global model weights.
    """
    initial_weights = initialize_global_model()
    return jsonify({"global_weights": [w.tolist() for w in initial_weights]}), 200


if __name__ == "__main__":
    initialize_global_model()
    app.run(host="0.0.0.0", port=5003, debug=False)
