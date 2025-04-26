#### python
# filepath: /home/mas/proj/study/reconstruct4D/reconstruct4D/ext/unsupervised_detection/scripts/test_DAVIS2016_raw.py
import os
import subprocess
import logging
import reconstruct4D.ext.unsupervised_detection.scripts.download_util as download_util  # Import the download utility

# --- Fixed Parameters ---
LOG_LEVEL = logging.INFO
TEST_CROP = 0.9
TEST_TEMPORAL_SHIFT = 1
GENERATE_VISUALIZATION = True
# --- End Fixed Parameters ---

logging.basicConfig(level=LOG_LEVEL, format="[%(levelname)s] %(message)s")


def main():
    # --- Define Paths and URLs for DAVIS 2016 ---
    script_dir = os.path.dirname(os.path.realpath(__file__))
    base_dir = os.path.abspath(
        os.path.join(script_dir, "..")
    )  # Go up one level from scripts/
    download_dir = os.path.join(base_dir, "download")
    results_dir = os.path.join(base_dir, "results", "DAVIS2016")  # Define results dir

    # Dataset
    dataset_name = "DAVIS2016"
    target_dataset_dir = os.path.join(download_dir, "DAVIS")
    dataset_download_url = (
        "https://graphics.ethz.ch/Downloads/Data/Davis/DAVIS-data.zip"
    )
    dataset_zip_base = "DAVIS-data"
    dataset_extracted_folder = "DAVIS"  # Verify this folder name inside the zip

    # Model Checkpoint
    model_ckpt_path = os.path.join(
        download_dir, "unsupervised_detection_models", "davis_best_model", "model.best"
    )
    model_ckpt_zip_url = "https://rpg.ifi.uzh.ch/data/unsupervised_detection_models.zip"
    model_ckpt_zip_filename = "unsupervised_detection_models.zip"

    # PWCNet Checkpoint
    pwc_ckpt_dir = os.path.join(
        download_dir, "pwcnet-lg-6-2-multisteps-chairsthingsmix"
    )
    pwc_ckpt_base = os.path.join(pwc_ckpt_dir, "pwcnet.ckpt-595000")
    pwc_gdown_folder_url = (
        "https://drive.google.com/drive/folders/1gtGx_6MjUQC5lZpl6-Ia718Y_0pvcYou"
    )

    # --- Ensure Prerequisites ---
    logging.info(f"--- Checking Prerequisites for {dataset_name} ---")
    os.makedirs(download_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)  # Ensure results dir exists

    # 1. Dataset
    if not download_util.ensure_dataset(
        dataset_name,
        target_dataset_dir,
        dataset_download_url,
        dataset_zip_base,
        dataset_extracted_folder,
        download_dir,
    ):
        logging.error(f"Failed to prepare dataset {dataset_name}. Exiting.")
        exit(1)

    # 2. Model Checkpoint
    if not download_util.ensure_model_checkpoint(
        model_ckpt_path, model_ckpt_zip_url, model_ckpt_zip_filename, download_dir
    ):
        logging.error(f"Failed to prepare model checkpoint {model_ckpt_path}. Exiting.")
        exit(1)

    # 3. PWCNet Checkpoint
    if not download_util.ensure_pwc_checkpoint(
        pwc_ckpt_base, pwc_gdown_folder_url, download_dir
    ):
        logging.error(f"Failed to prepare PWCNet checkpoint {pwc_ckpt_base}. Exiting.")
        exit(1)

    logging.info("--- Prerequisites Met ---")

    # --- Run Test Generator ---
    logging.info("Starting test generation...")
    test_command = [
        "python3",
        os.path.join(script_dir, "test_generator.py"),  # Path to test_generator.py
        f"--dataset={dataset_name}",
        f"--ckpt_file={model_ckpt_path}",
        f"--flow_ckpt={pwc_ckpt_base}",
        f"--test_crop={TEST_CROP}",
        f"--test_temporal_shift={TEST_TEMPORAL_SHIFT}",
        f"--root_dir={target_dataset_dir}",
        f"--test_save_dir={results_dir}",
    ]
    if GENERATE_VISUALIZATION:
        test_command.append("--generate_visualization")

    logging.info(f"Running command: {' '.join(test_command)}")
    try:
        # Run the command from the script's directory
        subprocess.run(test_command, check=True, cwd=script_dir)
        logging.info("Test generation finished successfully.")
    except subprocess.CalledProcessError as e:
        logging.error(f"Test generation failed with error code {e.returncode}.")
        exit(1)
    except FileNotFoundError:
        logging.error(
            f"Error: test_generator.py not found in {script_dir}. Make sure it exists."
        )
        exit(1)


if __name__ == "__main__":
    main()
