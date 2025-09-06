#!/bin/bash

docker build -f ci/Dockerfile.cifar10 src -t preprocess-cifar10:latest
docker build -f ci/Dockerfile.modelsave src -t cifar10-model-save:latest