import os
import subprocess
import logging
import download_util


def main():
    # --- Fixed Parameters ---
    dataset_name = "DAVIS2016"
    model_name = "Foels"
    dataset_download_urls = [
        "https://graphics.ethz.ch/Downloads/Data/Davis/DAVIS-data.zip",
    ]
    rootdidr_name = (
        dataset_name + "/DAVIS"
    )  # This is neccesary becuase DAVIS2016 zip top directory is DAVIS and unsupervised training needs root_dir as under DAVIS directory structure.

    LOG_LEVEL = 2
    TEST_CROP = 1.0
    TEST_TEMPORAL_SHIFT = 1
    GENERATE_VISUALIZATION = True
    # --- End Fixed Parameters ---

    logging.basicConfig(level=LOG_LEVEL, format="[%(levelname)s] %(message)s")

    # --- Define Paths and URLs ---
    script_dir = os.path.dirname(os.path.realpath(__file__))
    base_dir = os.path.abspath(
        os.path.join(script_dir, "..")
    )  # Go up one level from scripts/
    download_dir = os.path.join(base_dir, "download")
    results_dir = os.path.join(base_dir, "results", model_name, dataset_name)
    foels_resdir = os.path.join(base_dir, "../../..", "output", "davis")

    # --- Ensure Prerequisites ---
    logging.info(f"--- Checking Prerequisites for {dataset_name} ---")
    os.makedirs(download_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)  # Ensure results dir exists

    #  Dataset
    if not download_util.ensure_dataset(
        dataset_name,
        dataset_download_urls,
        download_dir,
    ):
        logging.error(f"Failed to prepare dataset {dataset_name}. Exiting.")
        exit(1)

    logging.info("--- Prerequisites Met ---")

    # --- Run Test Generator ---
    logging.info("Starting test generation...")
    test_command = [
        "python3",
        os.path.join(base_dir, "test_movobjextractor.py"),  # Path to test_generator.py
        f"--dataset={dataset_name}",
        "--batch_size=1",
        f"--test_crop={TEST_CROP}",
        f"--test_temporal_shift={TEST_TEMPORAL_SHIFT}",
        f"--root_dir={download_dir}/{rootdidr_name}",
        f"--test_save_dir={results_dir}",
        f"--foels_resdir={foels_resdir}",
        f"--log_level={LOG_LEVEL}",
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
            f"Error: test_movobjextractor.py not found in {script_dir}. Make sure it exists."
        )
        exit(1)


if __name__ == "__main__":
    main()
