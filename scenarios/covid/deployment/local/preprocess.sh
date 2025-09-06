#!/bin/bash

export REPO_ROOT="$(git rev-parse --show-toplevel)"
export SCENARIO="covid"
export DATA_DIR=$REPO_ROOT/scenarios/$SCENARIO/data
export ICMR_INPUT_PATH=$DATA_DIR/icmr
export ICMR_OUTPUT_PATH=$DATA_DIR/icmr/preprocessed
export INDEX_INPUT_PATH=$DATA_DIR/index
export INDEX_OUTPUT_PATH=$DATA_DIR/index/preprocessed
export COWIN_INPUT_PATH=$DATA_DIR/cowin
export COWIN_OUTPUT_PATH=$DATA_DIR/cowin/preprocessed
docker compose -f docker-compose-preprocess.yml up --remove-orphans
