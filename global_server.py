from flask import Flask, request, jsonify, render_template
import numpy as np
import requests
from concurrent.futures import ThreadPoolExecutor
import threading
from datetime import datetime

app = Flask(__name__)

# Global Variables
received_weights = {}
registered_nodes = {}
node_schemas = {}
global_schema = None
current_round = 0
max_rounds = 5  # Set the maximum number of rounds
registered_nodes_info = {}  # Key: node_id, Value: {'webhook_url': ..., 'last_updated': ..., 'metrics': [...], 'logs': [...]}


@app.route('/health', methods=['GET'])
def health():
    return 'OK', 200


def safe_format(value, decimal_places=4):
    try:
        return f"{float(value):.{decimal_places}f}"
    except (TypeError, ValueError):
        return "N/A"


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
    registered_nodes_info[node_id] = {
        'webhook_url': webhook_url,
        'last_updated': None,
        'metrics': [],
        'logs': []
    }
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
    global received_weights, current_round
    data = request.json
    node_id = data['node_id']
    weights = data['weights']

    print(f"Received weights from {node_id}.")
    received_weights[node_id] = [np.array(w) for w in weights]

    # Check if all registered nodes have submitted weights
    if len(received_weights) == len(registered_nodes):
        current_round += 1
        print(f"All nodes have submitted weights for round {current_round}.")
        if current_round > max_rounds:
            print("Maximum number of rounds reached. Stopping aggregation.")
            # Optionally notify nodes that training is complete
            notify_training_complete()
            return jsonify({"message": "Training complete"}), 200

        print("Aggregating weights...")
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


def notify_training_complete():
    """
    Notify hospital nodes that training is complete.
    """

    def notify_node(node_id, url):
        try:
            response = requests.post(
                f'{url}/training_complete',
                json={"message": "Training complete"}
            )
            if response.status_code == 200:
                print(f"Successfully notified {node_id} of training completion.")
            else:
                print(f"Failed to notify {node_id}: {response.status_code} {response.text}")
        except Exception as e:
            print(f"Error notifying {node_id}: {e}")

    with ThreadPoolExecutor() as executor:
        for node_id, webhook_url in registered_nodes.items():
            executor.submit(notify_node, node_id, webhook_url)


@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html', nodes=registered_nodes_info)


@app.route('/submit_metrics', methods=['POST'])
def submit_metrics():
    data = request.json
    node_id = data['node_id']
    metrics = data['metrics']
    metrics['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    node_info = registered_nodes_info.get(node_id)
    if node_info:
        node_info['metrics'].append(metrics)
        node_info['last_updated'] = metrics['timestamp']
        print(f"Received metrics from {node_id} for round {metrics['round']}.")
        return jsonify({"message": "Metrics received"}), 200
    else:
        return jsonify({"message": "Node not registered"}), 400


@app.route('/node/<node_id>')
def node_details(node_id):
    node_info = registered_nodes_info.get(node_id)
    if not node_info:
        return "Node not found", 404

    metrics = node_info.get('metrics', [])
    logs = node_info.get('logs', [])
    last_updated = node_info.get('last_updated')

    return render_template('node_details.html',
                           node_id=node_id,
                           metrics=metrics,
                           logs=logs,
                           last_updated=last_updated)


@app.route('/submit_logs', methods=['POST'])
def submit_logs():
    data = request.json
    node_id = data['node_id']
    log_line = data['log']

    node_info = registered_nodes_info.get(node_id)
    if node_info:
        node_info['logs'].append(log_line)
        return jsonify({"message": "Log received"}), 200
    else:
        return jsonify({"message": "Node not registered"}), 400


if __name__ == "__main__":
    app.jinja_env.filters['safe_format'] = safe_format
    app.run(host="0.0.0.0", port=5003, debug=False)
