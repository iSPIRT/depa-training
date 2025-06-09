#!/bin/bash

docker build -f ci/Dockerfile.bratsA src -t preprocess-brats-a:latest
docker build -f ci/Dockerfile.bratsB src -t preprocess-brats-b:latest
docker build -f ci/Dockerfile.bratsC src -t preprocess-brats-c:latest
docker build -f ci/Dockerfile.bratsD src -t preprocess-brats-d:latest
docker build -f ci/Dockerfile.modelsave src -t ccr-model-save:latest
