#!/bin/bash

export REPO_ROOT="$(git rev-parse --show-toplevel)"
export SCENARIO=brats
export DATA_DIR=$REPO_ROOT/scenarios/$SCENARIO/data

tar -xzf $DATA_DIR/brats_A.tar.gz -C $DATA_DIR/
tar -xzf $DATA_DIR/brats_B.tar.gz -C $DATA_DIR/
tar -xzf $DATA_DIR/brats_C.tar.gz -C $DATA_DIR/
tar -xzf $DATA_DIR/brats_D.tar.gz -C $DATA_DIR/

export BRATS_A_INPUT_PATH=$DATA_DIR/brats_A/
export BRATS_A_OUTPUT_PATH=$DATA_DIR/brats_A/preprocessed
export BRATS_B_INPUT_PATH=$DATA_DIR/brats_B/
export BRATS_B_OUTPUT_PATH=$DATA_DIR/brats_B/preprocessed
export BRATS_C_INPUT_PATH=$DATA_DIR/brats_C/
export BRATS_C_OUTPUT_PATH=$DATA_DIR/brats_C/preprocessed
export BRATS_D_INPUT_PATH=$DATA_DIR/brats_D/
export BRATS_D_OUTPUT_PATH=$DATA_DIR/brats_D/preprocessed

docker compose -f docker-compose-preprocess.yml up --remove-orphans
