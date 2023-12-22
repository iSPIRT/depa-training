#!/bin/bash

docker build -f ci/Dockerfile.preprocess src -t depa-mnist-preprocess:latest
docker build -f ci/Dockerfile.savemodel src -t depa-mnist-save-model:latest
