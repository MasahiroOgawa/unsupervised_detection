#!/bin/bash
#
# Script to compute raw results (before post-processing)
###

SCRIPT_DIR=$(dirname "$(realpath "$0")")

# parameters
DOWNLOAD_DIR="${SCRIPT_DIR}/../download"
DATASET_FILE="${DOWNLOAD_DIR}/DAVIS"
FOELS_RESDIR="${SCRIPT_DIR}/../../../../output/davis"
RESULT_DIR="${SCRIPT_DIR}/../results/Foels/DAVIS2016"
 # LOG_LEVEL=0: no log but save the result images, 1: print log, 2: display image
 # 3: debug with detailed image but without stopping, 4: slow (1min/frame) debug image
LOG_LEVEL=2


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

echo "[INFO] set conda env"
# deactivate uv venv first. otherwise, conda env will be hide.
deactivate
eval "$(conda shell.bash activate contextual-information-separation)"
echo "[INFO] env: $CONDA_DEFAULT_ENV"

echo "[INFO] start running a test..."
mkdir -p ${RESULT_DIR}
python3 ${SCRIPT_DIR}/../test_movobjextractor.py \
--dataset=DAVIS2016 \
--batch_size=1 \
--test_crop=1.0 \
--test_temporal_shift=1 \
--root_dir=${DATASET_FILE} \
--generate_visualization=True \
--test_save_dir=${RESULT_DIR} \
--foels_resdir=${FOELS_RESDIR} \
--log_level=${LOG_LEVEL}
echo "[INFO] finished the test."
