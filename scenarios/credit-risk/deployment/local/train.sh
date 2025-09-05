#!/bin/bash

export REPO_ROOT="$(git rev-parse --show-toplevel)"
export SCENARIO="credit-risk"

export DATA_DIR=$REPO_ROOT/scenarios/$SCENARIO/data
export MODEL_DIR=$REPO_ROOT/scenarios/$SCENARIO/modeller

export BANK_A_INPUT_PATH=$DATA_DIR/bank_a/preprocessed
export BANK_B_INPUT_PATH=$DATA_DIR/bank_b/preprocessed
export BUREAU_INPUT_PATH=$DATA_DIR/bureau/preprocessed
export FINTECH_INPUT_PATH=$DATA_DIR/fintech/preprocessed

export MODEL_OUTPUT_PATH=$MODEL_DIR/output
sudo rm -rf $MODEL_OUTPUT_PATH
mkdir -p $MODEL_OUTPUT_PATH 

export CONFIGURATION_PATH=$REPO_ROOT/scenarios/$SCENARIO/config

# Run consolidate_pipeline.sh to create pipeline_config.json
$REPO_ROOT/scenarios/$SCENARIO/config/consolidate_pipeline.sh

docker compose -f docker-compose-train.yml up --remove-orphans
