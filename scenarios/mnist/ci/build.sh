#!/bin/bash

docker build -f ci/Dockerfile.mnist_1 src -t preprocess-mnist_1:latest
docker build -f ci/Dockerfile.mnist_2 src -t preprocess-mnist_2:latest
docker build -f ci/Dockerfile.mnist_3 src -t preprocess-mnist_3:latest
docker build -f ci/Dockerfile.modelsave src -t ccr-model-save:latest
