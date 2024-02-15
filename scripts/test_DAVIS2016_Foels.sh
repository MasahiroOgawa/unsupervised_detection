#!/bin/bash
#
# Script to compute raw results (before post-processing)
###

SCRIPT_DIR=$(dirname "$(realpath "$0")")

# parameters
DOWNLOAD_DIR="${SCRIPT_DIR}/../download"
DATASET_FILE="${DOWNLOAD_DIR}/DAVIS"
RESULT_DIR="${SCRIPT_DIR}/../results/Foels/DAVIS2016"


echo "[INFO] start downloading data..."
mkdir -p ${DOWNLOAD_DIR}
(
    cd ${DOWNLOAD_DIR}   
    if [ ! -e ${DATASET_FILE} ]; then
	echo "[INFO] no DAVIS data found. start downloading it."
	wget https://graphics.ethz.ch/Downloads/Data/Davis/DAVIS-data.zip
	unzip DAVIS-data.zip
	rm DAVIS-data.zip
    fi
)
echo "[INFO] finished downloading."


echo "[INFO] start running a test..."
mkdir -p ${RESULT_DIR}
python3 test_movobjextractor.py \
--dataset=DAVIS2016 \
--batch_size=1 \
--test_crop=0.9 \
--test_temporal_shift=1 \
--root_dir=$DATASET_FILE \
--generate_visualization=True \
--test_save_dir=${RESULT_DIR}
echo "[INFO] finished the test."
