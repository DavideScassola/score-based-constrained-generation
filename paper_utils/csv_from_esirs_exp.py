import csv
import json
import os

# Specify the directory where the report files are stored
directory = "/home/davide/private-generative-experiments/artifacts/constrained_generation/eSIRS_bridging"

# Create a list to store the data
data = []

# Walk through the directory
for root, dirs, files in os.walk(directory):
    for file in files:
        # Check if the file is a JSON file
        if file == "plot_stats.json":
            # Construct the full path to the file
            file_path = os.path.join(root, file)

            # Open the JSON file and load the data
            with open(file_path, "r") as f:
                json_data = json.load(f)

            # Extract the folder name and the required values
            folder_name = os.path.basename(root)
            mean = round(json_data.get("mean(L)", "N/A"), 3)
            median = round(json_data.get("median(L)", "N/A"), 3)
            max_value = round(json_data.get("max(L)", "N/A"), 3)

            # Add the data to the list
            data.append([folder_name, mean, median, max_value])

# Specify the name of the CSV file
csv_file = "output.csv"

# Write the data to the CSV file
with open(csv_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Experiment", "Mean", "Median", "Max"])  # Write the header
    writer.writerows(data)  # Write the data
