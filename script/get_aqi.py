import os
import pandas as pd

# Define target folder and output file
target_folder = "./query/"
output_file = "consolidated_output.csv"

# List all CSV files in the folder
csv_files = [f for f in os.listdir(target_folder) if f.endswith('.csv')]

# Initialize an empty list to store DataFrames
dataframes = []

# Read each file starting from row 8
for file in csv_files:
    file_path = os.path.join(target_folder, file)
    df = pd.read_csv(file_path, skiprows=7)  # Skip first 7 rows

    # Forward fill the 'DATE' column with the latest non-null value
    if 'Date' in df.columns:
        df['Date'] = df['Date'].ffill()

    # Remove '*' from all columns
    df = df.applymap(lambda x: str(x).replace('*', '') if isinstance(x, str) else x)
    if 'Hour' in df.columns:
        df = df[pd.to_numeric(df['Hour'], errors='coerce').notna()]


    dataframes.append(df)

# Concatenate all dataframes
consolidated_df = pd.concat(dataframes, ignore_index=True)

# Save to CSV
consolidated_df.to_csv(output_file, index=False)

print(f"Consolidation complete. File saved as {output_file}")
