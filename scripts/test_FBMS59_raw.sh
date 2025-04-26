#!/bin/bash
#
# Script to compute raw results (before post-processing) on the FBMS-59 dataset
# Reference dataset info: https://lmb.informatik.uni-freiburg.de/resources/datasets/

set -e
SCRIPT_DIR=$(dirname "$(realpath "$0")")

# parameters
DOWNLOAD_DIR="${SCRIPT_DIR}/../download"
CKPT_FILE="${DOWNLOAD_DIR}/unsupervised_detection_models/fbms_best_model/model.best"
PWC_CKPT_FILE="${DOWNLOAD_DIR}/pwcnet-lg-6-2-multisteps-chairsthingsmix/pwcnet.ckpt-595000.data-00000-of-00001"
DATASET_FILE="${DOWNLOAD_DIR}/FBMS59"
RESULT_DIR="${SCRIPT_DIR}/../results/FBMS59"
LOG_LEVEL=2

echo "[INFO] start downloading data..."
mkdir -p "${DOWNLOAD_DIR}"
(
    cd "${DOWNLOAD_DIR}"
    if [ ! -f "${CKPT_FILE}.data*" ]; then
        echo "[INFO] No checkpoint file found. Please place the FBMS model in ${CKPT_FILE}."
        exit 1
    fi
    if [ ! -f "${PWC_CKPT_FILE}" ]; then
        echo "[INFO] No pwc checkpoint file found. start downloading it."
        gdown --folder "https://drive.google.com/drive/folders/1gtGx_6MjUQC5lZpl6-Ia718Y_0pvcYou"
        if [ $? -ne 0 ]; then
            echo "[ERROR] Failed to download PWC checkpoint. Please check your internet connection."
            exit 1
        fi
    fi
    if [ ! -e "${DATASET_FILE}" ]; then
        echo "[INFO] No FBMS-59 data found. Attempting to download..."
        wget https://lmb.informatik.uni-freiburg.de/resources/datasets/fbms/FBMS_Trainingset_large.zip -O FBMS-59.zip
        if [ $? -ne 0 ]; then
            echo "[ERROR] Failed to download FBMS-59 dataset. Please check your internet connection."
            rm -f FBMS-59.zip # Clean up the incomplete download
            exit 1
        fi
        echo "[INFO] Unzipping FBMS-59 dataset..."
        unzip FBMS-59.zip
        # Adjust the folder name if needed
        mv FBMS-59 "${DATASET_FILE}"
        rm FBMS-59.zip
    fi
)
echo "[INFO] Finished dataset preparations."

echo "[INFO] Start running a test on FBMS-59..."
mkdir -p "${RESULT_DIR}"
python3 test_generator.py \
--dataset=FBMS59 \
--ckpt_file="${CKPT_FILE}" \
--flow_ckpt="${PWC_CKPT_FILE}" \
--test_crop=1.0 \
--test_temporal_shift=1 \
--root_dir="${DATASET_FILE}" \
--generate_visualization=True \
--test_save_dir="${RESULT_DIR}" \
--log_level="${LOG_LEVEL}"
echo "[INFO] Finished the test on FBMS-59."