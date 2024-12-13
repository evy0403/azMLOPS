import os
import argparse
import pandas as pd
import subprocess

# Argument parser for dynamic paths
parser = argparse.ArgumentParser()
parser.add_argument("--trainingdata", type=str, required=True,
                    help="Path to the dataset for training")
parser.add_argument("--outputfolder", type=str, default="Out",
                    help="Folder to save the preprocessed data")
args = parser.parse_args()


input_file = args.trainingdata
output_folder = args.outputfolder

# Ensure the output folder exists
os.makedirs(output_folder, exist_ok=True)

try:
    # Step 1: Load the dataset
    print(f"Loading dataset from: {input_file}")
    data = pd.read_csv(input_file)

    # Step 2: Preprocess the data (example: drop rows with missing values and standardize columns)
    print("Preprocessing the data...")
    cleaned_data = data.dropna()  # Dropping rows with missing values
    
    # Example: Renaming columns for consistency
    cleaned_data.columns = [col.strip().lower().replace(" ", "_") for col in cleaned_data.columns]
    
    # Example: Normalize numerical data (optional step)
    for col in cleaned_data.select_dtypes(include=['float64', 'int64']).columns:
        cleaned_data[col] = (cleaned_data[col] - cleaned_data[col].mean()) / cleaned_data[col].std()

    # Step 3: Save the preprocessed dataset
    output_file = os.path.join(output_folder, "preprocessed_data.csv")
    print(f"Saving preprocessed data to: {output_file}")
    cleaned_data.to_csv(output_file, index=False)

    print("Data preprocessing completed successfully.")

    # Step 4: Push to GitHub
    print("Pushing preprocessed data to GitHub...")
    commands = [
        "git add .",
        f"git commit -m 'Added preprocessed data'",
        "git push"
    ]
    for command in commands:
        result = subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(result.stdout.decode().strip())

    print("Data pushed to GitHub successfully.")

except FileNotFoundError as e:
    print(f"Error: {e}. Please ensure the input file exists at {input_file}.")
except subprocess.CalledProcessError as e:
    print(f"Git command failed: {e.stderr.decode().strip()}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
