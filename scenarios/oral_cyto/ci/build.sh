#!/bin/bash
docker build -f ci/Dockerfile.oralcytoA src -t preprocess-oral-cyto-a:latest
docker build -f ci/Dockerfile.oralcytoB src -t preprocess-oral-cyto-b:latest
docker build -f ci/Dockerfile.modelsave src -t oral-cyto-model-save:latest
