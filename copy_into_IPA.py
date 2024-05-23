import os
import shutil
import yaml

# Define the dataset paths
source_path = "IPA/cheese_path.yaml"

# Define the destination base path
destination_base_path = "/users/eleves-b/2022/edouard.rabasse/Documents/INF473V-challenge-1/dataset/IPA_training"
with open(source_path, 'r') as file:
    cheese_data = yaml.safe_load(file)

# Create the directories and copy the files
for cheese, paths in cheese_data.items():
    destination_path = os.path.join(destination_base_path, cheese)
    os.makedirs(destination_path, exist_ok=True)
    for path in paths:
        destination_file_path = os.path.join(destination_path, os.path.basename(path))
        if os.path.isfile(path) and not os.path.isfile(destination_file_path):
            shutil.copy(path, destination_path)
            print(f"Copied {path} to {destination_path}")
        elif not os.path.isfile(path):
            print(f"File not found: {path}")
        else:
            print(f"File already exists: {destination_file_path}")