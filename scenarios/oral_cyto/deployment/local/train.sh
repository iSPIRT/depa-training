#!/bin/bash

export REPO_ROOT="$(git rev-parse --show-toplevel)"
export SCENARIO=oral_cyto
export DATA_DIR=$REPO_ROOT/scenarios/$SCENARIO/data
export MODEL_DIR=$REPO_ROOT/scenarios/$SCENARIO/modeller

export ORAL_CYTO_A_INPUT_PATH=$DATA_DIR/oral_cyto_A/preprocessed
export ORAL_CYTO_B_INPUT_PATH=$DATA_DIR/oral_cyto_B/preprocessed

export MODEL_INPUT_PATH=$MODEL_DIR/models

export MODEL_OUTPUT_PATH=$MODEL_DIR/output
sudo rm -rf $MODEL_OUTPUT_PATH
mkdir -p $MODEL_OUTPUT_PATH

export CONFIGURATION_PATH=$REPO_ROOT/scenarios/$SCENARIO/config

# Run consolidate_pipeline.sh to create pipeline_config.json
$REPO_ROOT/scenarios/$SCENARIO/config/consolidate_pipeline.sh

docker compose -f docker-compose-train.yml up --remove-orphans
