#!/bin/bash

docker build -f ci/Dockerfile.medqa src -t preprocess-medqa:latest
docker build -f ci/Dockerfile.chatdoctor src -t preprocess-chatdoctor:latest
docker build -f ci/Dockerfile.medquad src -t preprocess-medquad:latest
docker build -f ci/Dockerfile.modelsave src -t ccr-model-save:latest