#!/bin/bash
export REPO_ROOT="$(git rev-parse --show-toplevel)"
export SCENARIO="mule-detection"
export DATA_DIR=$REPO_ROOT/scenarios/$SCENARIO/data
export BANK_1_INPUT_PATH=$DATA_DIR/bank_1
export BANK_1_OUTPUT_PATH=$DATA_DIR/bank_1/preprocessed
export BANK_2_INPUT_PATH=$DATA_DIR/bank_2
export BANK_2_OUTPUT_PATH=$DATA_DIR/bank_2/preprocessed
export BANK_3_INPUT_PATH=$DATA_DIR/bank_3
export BANK_3_OUTPUT_PATH=$DATA_DIR/bank_3/preprocessed
mkdir -p $BANK_1_OUTPUT_PATH
mkdir -p $BANK_2_OUTPUT_PATH
mkdir -p $BANK_3_OUTPUT_PATH
docker compose -f docker-compose-preprocess.yml up --remove-orphans
