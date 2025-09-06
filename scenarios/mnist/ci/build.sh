#!/bin/bash

docker build -f ci/Dockerfile.mnist src -t preprocess-mnist:latest
docker build -f ci/Dockerfile.modelsave src -t mnist-model-save:latest