#!/bin/bash

export REPO_ROOT="$(git rev-parse --show-toplevel)"
export SCENARIO="credit-risk"
export DATA_DIR=$REPO_ROOT/scenarios/$SCENARIO/data
export BANK_A_INPUT_PATH=$DATA_DIR/bank_a
export BANK_B_INPUT_PATH=$DATA_DIR/bank_b
export BUREAU_INPUT_PATH=$DATA_DIR/bureau
export FINTECH_INPUT_PATH=$DATA_DIR/fintech
export BANK_A_OUTPUT_PATH=$DATA_DIR/bank_a/preprocessed
export BANK_B_OUTPUT_PATH=$DATA_DIR/bank_b/preprocessed
export BUREAU_OUTPUT_PATH=$DATA_DIR/bureau/preprocessed
export FINTECH_OUTPUT_PATH=$DATA_DIR/fintech/preprocessed
mkdir -p $BANK_A_OUTPUT_PATH
mkdir -p $BANK_B_OUTPUT_PATH
mkdir -p $BUREAU_OUTPUT_PATH
mkdir -p $FINTECH_OUTPUT_PATH
docker compose -f docker-compose-preprocess.yml up --remove-orphans
