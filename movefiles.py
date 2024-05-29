import os
import shutil

def move_files_with_rename(src_dir, dest_dir):
    # Ensure destination directory exists
    os.makedirs(dest_dir, exist_ok=True)

    # Get list of files in the source directory
    files = os.listdir(src_dir)

    for file_name in files:
        # Construct full file path
        src_file_path = os.path.join(src_dir, file_name)
        dest_file_path = os.path.join(dest_dir, file_name)

        # If file already exists in the destination directory, rename it
        if os.path.exists(dest_file_path):
            base, extension = os.path.splitext(file_name)
            counter = 1
            new_file_name = f"{base}_{counter}{extension}"
            new_dest_file_path = os.path.join(dest_dir, new_file_name)

            # Increment the counter until a unique file name is found
            while os.path.exists(new_dest_file_path):
                counter += 1
                new_file_name = f"{base}_{counter}{extension}"
                new_dest_file_path = os.path.join(dest_dir, new_file_name)
            
            dest_file_path = new_dest_file_path

        # # Move the file
        # shutil.move(src_file_path, dest_file_path)
        # print(f"Moved '{src_file_path}' to '{dest_file_path}'")

        # Copy the file
        shutil.copy(src_file_path, dest_file_path)
        print(f"Copied '{src_file_path}' to '{dest_file_path}'")

if __name__ == "__main__":
    with open ('list_of_cheese.txt', 'r') as file:
        for line in file:
            line = line.strip()
            src_directory = f"dataset/train/IPA5/{line}"
            dest_directory = f"dataset/train/contexts_prompts2/{line}"
            move_files_with_rename(src_directory, dest_directory)
        print('done')
