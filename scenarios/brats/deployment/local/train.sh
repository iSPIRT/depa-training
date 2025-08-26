#!/bin/bash

export REPO_ROOT="$(git rev-parse --show-toplevel)"
export SCENARIO=brats
export DATA_DIR=$REPO_ROOT/scenarios/$SCENARIO/data
export MODEL_DIR=$REPO_ROOT/scenarios/$SCENARIO/modeller

export BRATS_A_INPUT_PATH=$DATA_DIR/brats_A/preprocessed
export BRATS_B_INPUT_PATH=$DATA_DIR/brats_B/preprocessed
export BRATS_C_INPUT_PATH=$DATA_DIR/brats_C/preprocessed
export BRATS_D_INPUT_PATH=$DATA_DIR/brats_D/preprocessed

export MODEL_INPUT_PATH=$MODEL_DIR/models

# export MODEL_OUTPUT_PATH=/tmp/output
export MODEL_OUTPUT_PATH=$MODEL_DIR/output
rm -rf $MODEL_OUTPUT_PATH
mkdir -p $MODEL_OUTPUT_PATH 

# export CONFIGURATION_PATH=/tmp
export CONFIGURATION_PATH=$REPO_ROOT/scenarios/$SCENARIO/config
# cp $PWD/../../config/pipeline_config.json /tmp/pipeline_config.json

# Run consolidate_pipeline.sh to create pipeline_config.json
$REPO_ROOT/scenarios/$SCENARIO/config/consolidate_pipeline.sh

docker compose -f docker-compose-train.yml up --remove-orphans
