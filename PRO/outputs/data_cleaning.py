import os
import argparse
import pandas as pd

# Argument parser to dynamically provide paths
parser = argparse.ArgumentParser()
parser.add_argument("--trainingdata", type=str, required=True,
                    help="Path to the dataset for training")
parser.add_argument("--outputmodel", type=str, required=True,
                    help="Directory to save the cleaned data")
args = parser.parse_args()

# Paths from arguments
input_file = args.trainingdata
output_folder = args.outputmodel
output_file = os.path.join(output_folder, "cleaned_data.csv")

# Ensure the output directory exists
os.makedirs(output_folder, exist_ok=True)

try:
    # Load the dataset
    print(f"Loading dataset from: {input_file}")
    data = pd.read_csv(input_file)

    # Clean the data (example: drop rows with missing values)
    print("Cleaning the data...")
    cleaned_data = data.dropna()

    # Save the cleaned dataset
    print(f"Saving cleaned data to: {output_file}")
    cleaned_data.to_csv(output_file, index=False)

    print("Data cleaning completed successfully.")

except FileNotFoundError as e:
    print(f"Error: {e}. Please ensure the input file exists at {input_file}.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
