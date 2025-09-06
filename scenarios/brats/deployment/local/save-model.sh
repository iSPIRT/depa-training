#!/bin/bash

export REPO_ROOT="$(git rev-parse --show-toplevel)"
export SCENARIO=brats
export MODEL_OUTPUT_PATH=$REPO_ROOT/scenarios/$SCENARIO/modeller/models
mkdir -p $MODEL_OUTPUT_PATH
export MODEL_CONFIG_PATH=$REPO_ROOT/scenarios/$SCENARIO/config/model_config.json
docker compose -f docker-compose-modelsave.yml up --remove-orphans