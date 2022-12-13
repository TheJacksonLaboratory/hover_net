#!/bin/bash

# Usage: bash run_wsi.sh < input dir holding one or more WSIs > < output dir > < image magnification level (20, 40) > <model type (pannuke, monusac) [default: pannuke] >
# For example:
# bash run_wsi.sh /mnt/data/zarr /mnt/data/result/ 40

SRC_DIR=$1
DST_DIR=$2
PROC_MAG=$3

if [ "$SRC_DIR" == "" ]; then
    echo "Source directory containing at least one WSI image is needed"
    exit 1
fi

if [ "$DST_DIR" == "" ]; then
    echo "Destination directory where to store the inference output is needed"
    exit 1
fi

if [ "$PROC_MAG" == "" ]; then
    PROC_MAG=40
fi

AMB_SIZE=""
TILE_SHAPE=""
PATCH_SIZE=""
MODEL_TYPE=""

while getopts ":m:p:t:a:" opt; do
  case $opt in
    a)
      AMB_SIZE=$OPTARG
      ;;
    t)
      TILE_SHAPE=$OPTARG
      ;;
    p)
      PATCH_SIZE=$OPTARG
      ;;
    m)
      MODEL_TYPE=$OPTARG
      ;;
    \?)
      echo "Invalid option: -$OPTARG" >&2
      exit 1
      ;;
    :)
      echo "Option -$OPTARG requires an argument." >&2
      exit 1
      ;;
  esac
done

if [ "$PATCH_SIZE" != "" ]; then
    PATCH_SIZE="--patch_input_size ${PATCH_SIZE}"
fi

if [ "$TILE_SHAPE" != "" ]; then
    TILE_SHAPE="--tile_shape ${TILE_SHAPE}"
fi

if [ "$AMB_SIZE" != "" ]; then
    AMB_SIZE="--ambiguous_size=${AMB_SIZE}"
fi

if [ "$MODEL_TYPE" == "" ]; then
    MODEL_TYPE="pannuke"
fi

if [ "$CUDA_VISIBLE_DEVICES" == "" ]; then
    CUDA_VISIBLE_DEVICES="0"
fi

if [ "$MODEL_TYPE" == "pannuke" ]; then
    NR_TYPES=6
elif [ "$MODEL_TYPE" == "monusac" ]; then
    NR_TYPES=5
else
    echo "Model ${MODEL_TYPE} not supported."
    exit 1
fi

# If the destination directory does not exist, create it
if [ ! -d "${DST_DIR}" ]; then
    mkdir -p ${DST_DIR}
fi

run_command="python run_infer.py \
--gpu=${CUDA_VISIBLE_DEVICES} \
--nr_types=${NR_TYPES} \
--type_info_path=type_info.json \
--model_mode=fast \
--model_path=pretrained/hovernet_fast_${MODEL_TYPE}_type_tf2pytorch.tar \
  ${PATCH_SIZE} \
wsi \
--proc_mag=${PROC_MAG} \
--input_dir=${SRC_DIR} \
--output_dir=${DST_DIR} \
--input_mask_dir=${DST_DIR}/mask/ \
${TILE_SHAPE} \
${AMB_SIZE} \
--save_thumb \
--save_mask"

echo $run_command
exec $run_command
