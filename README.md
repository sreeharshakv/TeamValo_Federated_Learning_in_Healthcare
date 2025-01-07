# Improving Federated Learning in Healthcare
### A Comparative Study of Synchronous Aggregation Using Patient Readmission Data

---

## Overview
This project focuses on enhancing **Federated Learning (FL)** frameworks for healthcare applications. It proposes a **synchronous aggregation method** to address challenges in privacy-preserving machine learning across distributed hospital networks. The framework demonstrates improved accuracy and stability for predicting patient readmissions while ensuring compliance with privacy regulations like **HIPAA**.

Key innovations include:
- **Pull-based learning** for efficiency and synchronization.
- **Push-based updates** to maintain consistent global model performance.
- A **fallback mechanism** to handle client participation variability.
- Addressing **non-i.i.d. (heterogeneous)** data distributions with adaptive weighting.

---

## Features
- **Privacy-Preserving Machine Learning**: Ensures data remains localized, complying with healthcare privacy laws.
- **Synchronous Aggregation**: Improves model convergence and stability across distributed hospital nodes.
- **Adaptive Weighting**: Balances contributions from nodes with varying dataset sizes.
- **Fallback Mechanism**: Ensures robustness when nodes fail to participate in a training round.

---

## Project Architecture
The project is implemented using Python and consists of the following key components:

1. **Global Server**:
   - Coordinates the aggregation process.
   - Distributes the global model weights to hospital nodes.
   - Implements fallback mechanisms for missing updates.

2. **Hospital Nodes**:
   - Train local models using partitioned datasets.
   - Submit model updates to the global server.

3. **Datasets**:
   - **Diabetes 130-US Hospitals dataset** is used to simulate real-world hospital data, partitioned into 5 hospital nodes.

---

## System Requirements
### Dependencies
- Python 3.7 or later
- Required libraries (see `requirements.txt`):
  - `numpy`
  - `pandas`
  - `tensorflow`
  - `flask`
  - `scikit-learn`
  - `matplotlib`
  - `seaborn`
- Docker for containerized deployment.
