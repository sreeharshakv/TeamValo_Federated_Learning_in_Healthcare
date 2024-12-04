import os
import pandas as pd
import numpy as np
import re


# Helper function to convert age range to numeric midpoint
def convert_age_to_numeric(age):
    match = re.match(r"\[(\d+)-(\d+)\)", age)
    if match:
        lower, upper = map(int, match.groups())
        return (lower + upper) // 2  # Integer midpoint
    return None


def generate_test_data():
    # Directory and file setup
    dataset_dir = "dataset_diabetes"
    data_file = "dataset_diabetes.zip"
    file_path = os.path.join(dataset_dir, "diabetic_data.csv")

    # Step 1: Download and extract the dataset
    if not os.path.exists(file_path):
        print("Dataset not found. Downloading...")
        if not os.path.exists(data_file):
            # Use curl to download if wget is unavailable
            os.system(
                f"curl -o {data_file} https://archive.ics.uci.edu/ml/machine-learning-databases/00296/dataset_diabetes.zip")
        os.system(f"unzip {data_file}")

    # Step 2: Load the dataset
    if os.path.exists(file_path):
        print("Loading dataset...")
        converters = {
            "age": convert_age_to_numeric,  # Convert age ranges to numeric midpoints
        }
        data = pd.read_csv(file_path, converters=converters)
    else:
        raise FileNotFoundError(f"Dataset not found at {file_path}. Please check the file path.")

    print(f"Total records in the dataset before cleaning: {len(data)}")

    # Step 3: Remove columns with the most missing values
    columns_to_drop = ["max_glu_serum", "A1Cresult"]
    print(f"\nRemoving columns with the most missing values: {columns_to_drop}")
    data.drop(columns=columns_to_drop, inplace=True)

    # Step 4: Clean the dataset
    print("\nCleaning the dataset...")
    data.replace("?", np.nan, inplace=True)  # Replace '?' with NaN
    data.drop(columns=["encounter_id", "patient_nbr", "payer_code", "weight"],
              inplace=True)  # Drop less relevant columns
    data.dropna(inplace=True)  # Remove rows with missing values

    print(f"Total records in the dataset after cleaning: {len(data)}")

    return data


def get_test_data(hospital_id, category):
    data = generate_test_data()

    print("\nPartitioning dataset by hospital category..")
    output_dir = "hospital_diabetes_data"
    os.makedirs(output_dir, exist_ok=True)

    subset = data.query(category)
    file_name = f"{hospital_id}_data.csv"
    file_path = os.path.join(output_dir, file_name)
    subset.to_csv(file_path, index=False)
    print(f"{hospital_id}: - Saved {len(subset)} records to {file_path}")

    return subset