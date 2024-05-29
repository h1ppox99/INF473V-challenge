import os

cheese_dir = '/users/eleves-a/2022/hippolyte.wallaert/Modal/INF473V-challenge/dataset/val'

# Function to remove .DS_Store files
def remove_ds_store(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file == '.DS_Store':
                file_path = os.path.join(root, file)
                os.remove(file_path)
                print(f"Removed: {file_path}")

# Function to process directories and skip symbolic links
def process_directory(directory):
    for entry in os.listdir(directory):
        entry_path = os.path.join(directory, entry)
        if os.path.isdir(entry_path) and not os.path.islink(entry_path):  # Check if the entry is a directory and not a symbolic link
            for image_name in os.listdir(entry_path):
                image_path = os.path.join(entry_path, image_name)
                if os.path.isfile(image_path):  # Check if the entry is a file
                    # Your processing code here
                    print(f"Processing {image_path}")
        else:
            print(f"Skipping non-directory or symbolic link entry: {entry_path}")

# Remove .DS_Store files in the main directory
remove_ds_store(cheese_dir)

# Process the directory
process_directory(cheese_dir)
