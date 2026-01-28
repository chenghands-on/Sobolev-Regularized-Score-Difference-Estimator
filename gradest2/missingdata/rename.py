import os

# Get the current working directory
current_directory = os.getcwd()

# List all files in the current directory
files = os.listdir(current_directory)

# Iterate over all files
for file_name in files:
    # Check if the file name starts with 'irir_imputed_' and ends with '.mat'
    if file_name.startswith("irir_imputed_") and file_name.endswith(".mat"):
        # Create the new file name by removing the 'irir_imputed_' prefix
        new_file_name = file_name.replace("irir_imputed_", "", 1)
        
        # Construct full file paths
        old_file_path = os.path.join(current_directory, file_name)
        new_file_path = os.path.join(current_directory, new_file_name)
        
        # Rename the file
        os.rename(old_file_path, new_file_path)
        print(f'Renamed: {old_file_path} -> {new_file_path}')

print("File renaming complete.")
