import os
import argparse
import pandas as pd

# Argument parser for dynamic paths
parser = argparse.ArgumentParser()
parser.add_argument("--trainingdata", type=str, required=True,
                    help="Path to the dataset for training")
parser.add_argument("--outputmodel", type=str, required=True,
                    help="Path to save the cleaned data (CSV file)")
args = parser.parse_args()

# Extracting arguments
input_file = args.trainingdata
output_file = args.outputmodel

# Ensure the output folder exists
output_folder = os.path.dirname(output_file)
os.makedirs(output_folder, exist_ok=True)

try:
    # Step 1: Load the dataset
    print(f"Loading dataset from: {input_file}")
    data = pd.read_csv(input_file)

    # Step 2: Clean the data (example: drop rows with missing values)
    print("Cleaning the data...")
    cleaned_data = data.dropna()

    # Step 3: Save the cleaned dataset
    print(f"Saving cleaned data to: {output_file}")
    cleaned_data.to_csv(output_file, index=False)

    print("Data cleaning completed successfully.")
except FileNotFoundError as e:
    print(f"Error: {e}. Please ensure the input file exists at {input_file}.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
