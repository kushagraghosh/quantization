import os
import yaml
import gdown
import zipfile
import shutil


def download_and_unzip(zip_url, output_folder='./Dataset',  folders_to_delete=None):
    # Extract the Google Drive ID from the URL
    file_id = zip_url.split('/')[-2]
    download_url = f"https://drive.google.com/uc?id={file_id}"

    # Check if download URL has public permissions set
    try:
        zip_file = "downloaded_zip_file.zip"
        gdown.download(download_url, zip_file, quiet=False)

        # Create output folder if it doesn't exist
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # Unzip the downloaded file
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(output_folder)

        print(f"Files extracted to '{output_folder}'")

        # Clean up the downloaded zip file
        os.remove(zip_file)

        # Clean up the Dataset folder
        if folders_to_delete:
            for folder in folders_to_delete:
                folder_path = os.path.join(output_folder, folder)
                if os.path.exists(folder_path) and os.path.isdir(folder_path):
                    shutil.rmtree(folder_path)
                    print(f"Deleted folder: {folder_path}")
                else:
                    print(f"Folder not found or not a directory: {folder_path}")

        # Load the existing data.yaml file
        data_yaml_path = '../Dataset/data.yaml'
        with open(data_yaml_path, 'r') as file:
            data_config = yaml.safe_load(file)

        # Update paths to be relative
        data_config['train'] = './images/train'
        data_config['val'] = './images/val'
        data_config['test'] = './images/test'

        # Write the updated content back to the data.yaml file with proper formatting for arrays
        with open(data_yaml_path, 'w') as file:
            yaml.dump(data_config, file, default_flow_style=None, sort_keys=False)

        print(f"Updated '{data_yaml_path}' with relative paths.")

    except gdown.exceptions.FileURLRetrievalError as e:
        print("Failed to retrieve file url. Please check the permissions on Google Drive and try again.")
        print(e)

# Example usage:
zip_url = "https://drive.google.com/file/d/1UknFvrQTOlIAxhkf3CHvbaSSsHru-fPP/view?usp=sharing"
download_and_unzip(zip_url, folders_to_delete=['.config', '.ipynb_checkpoints'])
