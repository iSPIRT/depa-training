#!/bin/bash

docker build -f ci/Dockerfile.icmr src -t preprocess-icmr:latest
docker build -f ci/Dockerfile.index src -t preprocess-index:latest
docker build -f ci/Dockerfile.cowin src -t preprocess-cowin:latest
docker build -f ci/Dockerfile.modelsave src -t covid-model-save:latest
