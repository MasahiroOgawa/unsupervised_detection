import os
import requests
import zipfile
import subprocess
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


def download_file(url, destination_path):
    """_summary_
    Downloads a file from a URL to a destination path.

    Args:
        url (_type_): _description_
        destination_path (_type_): _description_. full path of the destination file name.
    """
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()  # raise an exception for HTTP errors
        total_size = int(response.headers.get("content-length", 0))
        block_size = 1024  # 1 Kibibyte

        logging.info(f"Downloading from {url} to {destination_path}")
        with open(destination_path, "wb") as file:
            with tqdm(
                desc="Downloading file",
                total=total_size,
                unit="iB",
                unit_scale=True,
                unit_divisor=block_size,
            ) as bar:
                for data in response.iter_content(block_size):
                    size = file.write(data)
                    bar.update(size)

        if total_size != 0 and bar.n != total_size:
            logging.error("Something went wrong during download")
            return False

        logging.info(f"Downloaded successfully from {url}  to {destination_path}")
        return True

    except requests.exceptions.RequestException as e:
        logging.error(f"Failed to download {url}: {e}")
        if os.path.exists(destination_path):
            os.remove(destination_path)
        return False


def extract_zip(zip_path, extract_to_dir):
    """_summary_
    Extracts a zip file to a specified directory.

    Args:
        zip_path (_type_): _description_
        extract_to_dir (_type_): _description_
    """
    if not os.path.exists(zip_path):
        logging.error(f"Zip file {zip_path} does not exist.")
        return False

    try:
        logging.info(f"Downloading {zip_path} to {extract_to_dir}")
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


def run_gdown(folder_url, destination_dir):
    """_summary_
    Downloads a folder from Google Drive using gdown.

    Args:
        folder_url (_type_): _description_
        destination_dir (_type_): _description_. destination full path directory name.
    """
    try:
        logging.info(f"Downloading {folder_url} to {destination_dir}")
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
    download_urls,
    destination_dir,
):
    """
    Checks for dataset, downloads and extracts if missing.

    Args:
        dataset_name (str): Name of the dataset. (e.g. "FBMS")
        download_urls (str): URLs to download the dataset zip file. (e.g. ["https://example.com/dataset.zip","https://example.com/dataset2.zip"])
        destination_dir (str): Directory to save the dataset. (e.g. "/home/user/downloads/FBMS")
    """
    if os.path.exists(destination_dir) and os.listdir(destination_dir):
        logging.info(
            f"Dataset '{dataset_name}' already exists at {destination_dir}. Skipping download."
        )
        return True

    logging.info(
        f"Dataset '{dataset_name}' not found at {destination_dir}. Downloading..."
    )
    for download_url in download_urls:
        os.makedirs(destination_dir, exist_ok=True)
        zip_basefname = dataset_name
        zip_filepath = os.path.join(destination_dir, f"{zip_basefname}.zip")
        zip_extracted_path = os.path.join(destination_dir, dataset_name)

        if not download_file(download_url, zip_filepath):
            logging.error(
                f"Failed to download {dataset_name} zip file from {download_url}."
            )
            return False  # Stop if download fails

        if not extract_zip(zip_filepath, zip_extracted_path):
            logging.error(f"Failed to extract {zip_filepath} to {zip_extracted_path}.")
            return False  # Stop if extraction fails

        # Check if the extracted folder exists
        if not os.path.exists(zip_extracted_path):
            logging.error(
                f"Expected extracted folder '{zip_extracted_path}' not found after unzipping."
            )
            return False

        # Cleanup zip file
        try:
            logging.info(f"Cleaning up {zip_filepath}")
            os.remove(zip_filepath)
        except OSError as e:
            logging.warning(f"Could not remove zip file {zip_filepath}. Error: {e}")

    logging.info(f"Successfully prepared dataset '{dataset_name}'.")
    return True


def ensure_pwc_checkpoint(gdown_folder_url, pwc_ckpt_path):
    """Checks for PWCNet checkpoint files, downloads via gdown if missing."""
    # Check for one of the expected files (adjust extensions if needed)
    pwc_dir = os.path.dirname(pwc_ckpt_path)
    if os.path.exists(pwc_ckpt_path):
        logging.info(f"PWCNet checkpoint found at {pwc_dir}. Skipping download.")
        return True

    logging.info("PWCNet checkpoint not found. Attempting download via gdown...")
    # gdown downloads the *contents* of the folder into the target directory
    if not run_gdown(
        gdown_folder_url, pwc_dir, description="Downloading PWCNet checkpoint"
    ):
        return False

    # Verify again after download attempt
    if not os.path.exists(pwc_ckpt_path):
        logging.error(
            f"PWCNet checkpoint file {pwc_ckpt_path} still not found after gdown attempt."
        )
        return False

    logging.info("Successfully prepared PWCNet checkpoint.")
    return True


def ensure_model_checkpoint(download_url, ckpt_path, zip_path=None):
    """
    Checks for the specific model checkpoint, downloads and extracts parent zip if missing.
    Args:
        download_url (str): URL to download the zip file containing the model checkpoint.
        ckpt_path (str): Path to the specific model checkpoint file.
        zip_path (str): Path to the already downloaded zip file containing the model checkpoint. otherwise you don't need to set this."""
    if os.path.exists(ckpt_path):
        logging.info(f"Model checkpoint found: {ckpt_path}. Skipping download.")
        return True

    logging.info(f"Model checkpoint {ckpt_path} not found.")

    # Check if the zip exists first
    if not os.path.exists(zip_path):
        logging.info(f"Checkpoint archive {zip_path} not found. Downloading...")
        if not download_file(download_url, zip_path):
            return False  # Stop if download fails
    else:
        logging.info(f"Checkpoint archive {zip_path} found.")

    # Extract the archive (even if it existed, maybe extraction failed before)
    extract_target_dir = os.path.dirname(ckpt_path)
    if not extract_zip(zip_path, extract_target_dir):
        return False

    # Verify the specific checkpoint exists after extraction
    if not os.path.exists(ckpt_path):
        logging.error(
            f"Model checkpoint {ckpt_path} still not found after extracting {zip_path}."
        )
        logging.error(
            "Please check the contents of the zip file and the expected path."
        )
        return False

    # Clean up the zip file after successful extraction and verification
    try:
        os.remove(zip_path)
    except OSError as e:
        logging.warning(f"Could not remove checkpoint zip file {zip_path}. Error: {e}")

    logging.info(f"Successfully prepared model checkpoint {ckpt_path}.")
    return True
