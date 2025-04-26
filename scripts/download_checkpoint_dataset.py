import os
import requests
import zipfile
import shutil
import subprocess
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


def download_file(url, destination_path, description="Downloading file name"):
    """_summary_
    Downloads a file from a URL to a destination path.

    Args:
        url (_type_): _description_
        destination_path (_type_): _description_
        description (str, optional): _description_. Defaults to "Downloading file".
    """
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()  # raise an exception for HTTP errors
        total_size = int(response.headers.get("content-length", 0))
        block_size = 1024  # 1 Kibibyte

        logging.info(f"Downloading {description} from {url} to {destination_path}")
        with (
            open(destination_path, "wb") as file,
            tqdm(
                desc=description,
                total=total_size,
                unit="iB",
                unit_scale=True,
                unit_divisor=block_size,
            ) as bar,
        ):
            for data in response.iter_content(block_size):
                size = file.write(data)
                bar.update(len(size))

        if total_size != 0 and bar.n != total_size:
            logging.error("Something went wrong during download")
            return False

        logging.info(f"Downloaded {description} successfully")
        return True

    except requests.exceptions.RequestException as e:
        logging.error(f"Failed to download {url}: {e}")
        if os.path.exists(destination_path):
            os.remove(destination_path)
        return False


def extract_zip(zip_path, extract_to_dir, description="Extracting zip file"):
    """_summary_
    Extracts a zip file to a specified directory.

    Args:
        zip_path (_type_): _description_
        extract_to_dir (_type_): _description_
        description (str, optional): _description_. Defaults to "Extracting zip file".
    """
    if not os.path.exists(zip_path):
        logging.error(f"Zip file {zip_path} does not exist.")
        return False

    try:
        logging.info(f"{description}: {zip_path} to {extract_to_dir}")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_to_dir)
        logging.info(f"Successfully extracted {zip_path} to {extract_to_dir}")
        return True

    except zipfile.BadZipFile as e:
        logging.error(f"The file {zip_path} is not a zip or currupted: {e}")
        # Don't remove the zip file here, as user might want to retry.
        return False
    except Exception as e:
        logging.error(f"An unkown error occurred while extracting {zip_path}: {e}")
        # Don't remove the zip file here, as user might want to retry.
        return False


def run_gdown(folder_url, destination_dir, description="Downloading folder"):
    """_summary_
    Downloads a folder from Google Drive using gdown.

    Args:
        folder_url (_type_): _description_
        destination_dir (_type_): _description_
        description (str, optional): _description_. Defaults to "Downloading folder".
    """
    try:
        logging.info(f"{description}: {folder_url} to {destination_dir}")
        os.makedirs(destination_dir, exist_ok=True)
        subprocess.run(
            ["gdown", "--folder", folder_url, "-O", destination_dir], check=True
        )
        logging.info(f"Successfully downloaded {folder_url} to {destination_dir}")
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to download {folder_url}: {e}")
        return False
    except Exception as e:
        logging.error(f"An unknown error occurred while downloading {folder_url}: {e}")
        return False


def ensure_dataset(
    dataset_name,
    target_dir,
    download_url,
    zip_filename_base,
    extracted_dir,
    download_dir,
):
    """Checks for dataset, downloads and extracts if missing."""
    if os.path.exists(target_dir):
        logging.info(
            f"Dataset '{dataset_name}' already exists at {target_dir}. Skipping download."
        )
        return True

    logging.info(f"Dataset '{dataset_name}' not found at {target_dir}. Downloading...")
    os.makedirs(download_dir, exist_ok=True)
    zip_filepath = os.path.join(download_dir, f"{zip_filename_base}.zip")
    zip_extracted_path = os.path.join(download_dir, extracted_dir)

    if not download_file(
        download_url, zip_filepath, description=f"Downloading {dataset_name}"
    ):
        return False  # Stop if download fails

    if not extract_zip(
        zip_filepath, download_dir, description=f"Extracting {dataset_name}"
    ):
        if os.path.exists(zip_filepath):
            os.remove(zip_filepath)
        return False  # Stop if extraction fails

    # Move extracted folder to target directory
    if not os.path.exists(zip_extracted_path):
        logging.error(
            f"Expected extracted folder '{zip_extracted_path}' not found after unzipping."
        )
        if os.path.exists(zip_filepath):
            os.remove(zip_filepath)
        return False
    try:
        logging.info(f"Moving {zip_extracted_path} to {target_dir}")
        shutil.move(zip_extracted_path, target_dir)
    except Exception as e:
        logging.error(
            f"Failed to move {zip_extracted_path} to {target_dir}. Error: {e}"
        )
        # Cleanup
        if os.path.exists(zip_filepath):
            os.remove(zip_filepath)
        if os.path.exists(zip_extracted_path):
            shutil.rmtree(zip_extracted_path)  # Remove potentially partially moved dir
        return False

    # Cleanup zip file
    try:
        logging.info(f"Cleaning up {zip_filepath}")
        os.remove(zip_filepath)
    except OSError as e:
        logging.warning(f"Could not remove zip file {zip_filepath}. Error: {e}")

    logging.info(f"Successfully prepared dataset '{dataset_name}'.")
    return True


def ensure_pwc_checkpoint(pwc_ckpt_base_path, gdown_folder_url, download_dir):
    """Checks for PWCNet checkpoint files, downloads via gdown if missing."""
    # Check for one of the expected files (adjust extensions if needed)
    expected_file = f"{pwc_ckpt_base_path}.data-00000-of-00001"
    pwc_dir = os.path.dirname(pwc_ckpt_base_path)
    if os.path.exists(expected_file):
        logging.info(f"PWCNet checkpoint found at {pwc_dir}. Skipping download.")
        return True

    logging.info(f"PWCNet checkpoint not found. Attempting download via gdown...")
    # gdown downloads the *contents* of the folder into the target directory
    if not run_gdown(
        gdown_folder_url, pwc_dir, description="Downloading PWCNet checkpoint"
    ):
        return False

    # Verify again after download attempt
    if not os.path.exists(expected_file):
        logging.error(
            f"PWCNet checkpoint file {expected_file} still not found after gdown attempt."
        )
        return False

    logging.info("Successfully prepared PWCNet checkpoint.")
    return True


def ensure_model_checkpoint(ckpt_path, download_url, zip_filename, download_dir):
    """Checks for the specific model checkpoint, downloads and extracts parent zip if missing."""
    if os.path.exists(ckpt_path):
        logging.info(f"Model checkpoint found: {ckpt_path}. Skipping download.")
        return True

    logging.info(f"Model checkpoint {ckpt_path} not found.")
    # Assume the checkpoint is inside a zip file that needs downloading/extracting
    zip_filepath = os.path.join(download_dir, zip_filename)
    extract_target_dir = download_dir  # Extract to the main download dir

    # Check if the zip exists first
    if not os.path.exists(zip_filepath):
        logging.info(f"Checkpoint archive {zip_filename} not found. Downloading...")
        if not download_file(
            download_url,
            zip_filepath,
            description="Downloading model checkpoints archive",
        ):
            return False  # Stop if download fails
    else:
        logging.info(f"Checkpoint archive {zip_filename} found.")

    # Extract the archive (even if it existed, maybe extraction failed before)
    # We extract to the download_dir, assuming the zip contains the
    # 'unsupervised_detection_models' folder structure.
    if not extract_zip(
        zip_filepath, extract_target_dir, description="Extracting model checkpoints"
    ):
        return False

    # Verify the specific checkpoint exists after extraction
    if not os.path.exists(ckpt_path):
        logging.error(
            f"Model checkpoint {ckpt_path} still not found after extracting {zip_filename}."
        )
        logging.error(
            "Please check the contents of the zip file and the expected path."
        )
        return False

    # Clean up the zip file after successful extraction and verification
    try:
        os.remove(zip_filepath)
    except OSError as e:
        logging.warning(
            f"Could not remove checkpoint zip file {zip_filepath}. Error: {e}"
        )

    logging.info(f"Successfully prepared model checkpoint {ckpt_path}.")
    return True
