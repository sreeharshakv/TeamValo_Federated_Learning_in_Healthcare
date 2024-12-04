from flask import Flask, request, jsonify
import numpy as np
import requests
from concurrent.futures import ThreadPoolExecutor
import threading

app = Flask(__name__)

# Global Variables
received_weights = {}
registered_nodes = {}
node_schemas = {}
global_schema = None


@app.route('/health', methods=['GET'])
def health():
    return 'OK', 200


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


@app.route('/submit_schema', methods=['POST'])
def submit_schema():
    data = request.json
    node_id = data['node_id']
    schema = data['schema']
    node_schemas[node_id] = schema
    print(f"Received schema from node {node_id}.")

    # Check if all nodes have submitted their schemas
    if len(node_schemas) == len(registered_nodes):
        print("All node schemas received. Creating global schema...")
        create_global_schema()
        distribute_global_schema()
    return jsonify({"message": "Schema received"}), 200


def create_global_schema():
    global global_schema
    global_schema = {}
    for node_schema in node_schemas.values():
        for feature, categories in node_schema.items():
            if feature not in global_schema:
                global_schema[feature] = set(categories)
            else:
                global_schema[feature].update(categories)
    # Convert sets to sorted lists for consistent ordering
    for feature in global_schema:
        global_schema[feature] = sorted(global_schema[feature])
    print("Global schema created.")


def distribute_global_schema():
    print("Distributing global schema to all nodes...")
    for node_id, webhook_url in registered_nodes.items():
        threading.Thread(target=send_global_schema, args=(node_id, webhook_url)).start()


def send_global_schema(node_id, webhook_url):
    payload = {'global_schema': global_schema}
    try:
        response = requests.post(f"{webhook_url}/receive_global_schema", json=payload)
        if response.status_code == 200:
            print(f"Global schema sent to node {node_id}.")
        else:
            print(f"Failed to send global schema to node {node_id}: {response.text}")
    except Exception as e:
        print(f"Error sending global schema to node {node_id}: {e}")


def aggregate_weights(weights_list):
    """
    Perform Federated Averaging to aggregate weights.
    """
    print("Aggregating weights...")
    aggregated_weights = []
    for layer_weights in zip(*weights_list):  # Iterate layer by layer
        aggregated_layer = np.mean(np.array(layer_weights), axis=0)
        aggregated_weights.append(aggregated_layer)
    return aggregated_weights


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
    received_weights[node_id] = [np.array(w) for w in weights]

    # Check if all registered nodes have submitted weights
    if len(received_weights) == len(registered_nodes):
        print("All nodes have submitted weights. Aggregating...")
        aggregated_weights = aggregate_weights(list(received_weights.values()))
        received_weights = {}  # Clear buffer

        # Notify hospital nodes about new global weights
        notify_hospital_nodes(aggregated_weights)

    return jsonify({"message": "Weights received", "status": "success"}), 200


def notify_hospital_nodes(aggregated_weights):
    """
    Notify hospital nodes via their webhook endpoints using multithreading.
    """

    def notify_node(node_id, url):
        try:
            response = requests.post(
                f'{url}/notify_weights',
                json={"global_weights": [w.tolist() for w in aggregated_weights]}
            )
            if response.status_code == 200:
                print(f"Successfully notified {node_id}.")
            else:
                print(f"Failed to notify {node_id}: {response.status_code} {response.text}")
        except Exception as e:
            print(f"Error notifying {node_id}: {e}")

    with ThreadPoolExecutor() as executor:
        for node_id, webhook_url in registered_nodes.items():
            executor.submit(notify_node, node_id, webhook_url)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5003, debug=False)
