#!/bin/bash

export REPO_ROOT="$(git rev-parse --show-toplevel)"
export SCENARIO="covid"
export DATA_DIR=$REPO_ROOT/scenarios/$SCENARIO/data
export MODEL_DIR=$REPO_ROOT/scenarios/$SCENARIO/modeller

export ICMR_INPUT_PATH=$DATA_DIR/icmr/preprocessed
export INDEX_INPUT_PATH=$DATA_DIR/index/preprocessed
export COWIN_INPUT_PATH=$DATA_DIR/cowin/preprocessed

export MODEL_INPUT_PATH=$MODEL_DIR/models

# export MODEL_OUTPUT_PATH=/tmp/output
export MODEL_OUTPUT_PATH=$MODEL_DIR/output
rm -rf $MODEL_OUTPUT_PATH
mkdir -p $MODEL_OUTPUT_PATH

export CONFIGURATION_PATH=$REPO_ROOT/scenarios/$SCENARIO/config
# export CONFIGURATION_PATH=/tmp
# cp $PWD/../../config/pipeline_config.json /tmp/pipeline_config.json

# Run consolidate_pipeline.sh to create pipeline_config.json
$REPO_ROOT/scenarios/$SCENARIO/config/consolidate_pipeline.sh

docker compose -f docker-compose-train.yml up --remove-orphans
