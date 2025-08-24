#!/bin/bash

export REPO_ROOT="$(git rev-parse --show-toplevel)"
export SCENARIO="cifar10"
export DATA_DIR=$REPO_ROOT/scenarios/$SCENARIO/data
export CIFAR10_INPUT_PATH=$DATA_DIR
export CIFAR10_OUTPUT_PATH=$DATA_DIR/preprocessed
mkdir -p $CIFAR10_OUTPUT_PATH
docker compose -f docker-compose-preprocess.yml up --remove-orphans
