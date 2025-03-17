'''This script removes object scene instances with missing files, before data preprocessing.'''

import os
import glob

# Define the base directory
base_dir = '../Data/Jacquard_Dataset_1'

# List of file types
file_types = ['grasps.txt', 'RGB.png', 'stereo_depth.tiff', 'perfect_depth.tiff', 'mask.png']

# Navigate through each subdirectory
for dir_name in os.listdir(base_dir):
    # Construct the full directory path
    full_dir_name = os.path.join(base_dir, dir_name)
    
    if os.path.isdir(full_dir_name):
        # Get the unique object prefixes in the directory
        object_prefixes = set(fname.split('_')[0] for fname in os.listdir(full_dir_name))
        
        # Check each object
        for prefix in object_prefixes:
            # Flag to track if all files for the current object exist
            all_files_exist = True
            
            for file_type in file_types:
                # Construct the expected filename
                expected_file = f"{prefix}_{dir_name}_{file_type}"
                
                # Check if the file exists
                if not os.path.isfile(os.path.join(full_dir_name, expected_file)):
                    all_files_exist = False
                    break  # No need to check other files for this object
            
            # If not all files exist, delete all files for this object
            if not all_files_exist:
                for file_to_delete in glob.glob(os.path.join(full_dir_name, f"{prefix}_{dir_name}_*")):
                    os.remove(file_to_delete)

        # After checking all objects, if directory is empty, remove it
        if not os.listdir(full_dir_name):
            os.rmdir(full_dir_name)
