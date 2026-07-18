#!/bin/bash
export REPO_ROOT="$(git rev-parse --show-toplevel)"
export SCENARIO="mule-detection"
export DATA_DIR=$REPO_ROOT/scenarios/$SCENARIO/data
export MODEL_DIR=$REPO_ROOT/scenarios/$SCENARIO/modeller
export BANK_1_INPUT_PATH=$DATA_DIR/bank_1/preprocessed
export BANK_2_INPUT_PATH=$DATA_DIR/bank_2/preprocessed
export BANK_3_INPUT_PATH=$DATA_DIR/bank_3/preprocessed
export MODEL_OUTPUT_PATH=$MODEL_DIR/output
sudo rm -rf $MODEL_OUTPUT_PATH
mkdir -p $MODEL_OUTPUT_PATH
export CONFIGURATION_PATH=$REPO_ROOT/scenarios/$SCENARIO/config
$REPO_ROOT/scenarios/$SCENARIO/config/consolidate_pipeline.sh
docker compose -f docker-compose-train.yml up --remove-orphans
