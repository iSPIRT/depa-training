#!/bin/bash
export REPO_ROOT="$(git rev-parse --show-toplevel)"
export SCENARIO="oral_cyto"
export DATA_DIR=$REPO_ROOT/scenarios/$SCENARIO/data
export ORAL_CYTO_A_INPUT_PATH=$DATA_DIR/raw/oral_cytology_dataset
export ORAL_CYTO_A_OUTPUT_PATH=$DATA_DIR/oral_cyto_A/preprocessed
export ORAL_CYTO_B_INPUT_PATH=$DATA_DIR/raw/oral_cytology_dataset
export ORAL_CYTO_B_OUTPUT_PATH=$DATA_DIR/oral_cyto_B/preprocessed
mkdir -p $ORAL_CYTO_A_OUTPUT_PATH
mkdir -p $ORAL_CYTO_B_OUTPUT_PATH
docker compose -f docker-compose-preprocess.yml up --remove-orphans
