import os
import shutil

def sort_files(directory):
    # Paths for 'y' and 'n' folders
    y_folder = os.path.join(directory, 'y')
    n_folder = os.path.join(directory, 'n')

    # Create 'y' and 'n' folders if they don't exist
    if not os.path.exists(y_folder):
        os.makedirs(y_folder)
    if not os.path.exists(n_folder):
        os.makedirs(n_folder)

    # List all files in the directory
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)

        # Check if it's a file and not a directory
        if os.path.isfile(file_path):
            # Sort files starting with 'y' or 'n' into respective folders
            if filename.lower().startswith('y'):
                shutil.move(file_path, os.path.join(y_folder, filename))
            elif filename.lower().startswith('n'):
                shutil.move(file_path, os.path.join(n_folder, filename))

# Replace '/path/to/directory' with the path of your directory
sort_files('D:\Aqoustics\Boat\Y_N_Boat\All files')
