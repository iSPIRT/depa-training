#!/bin/bash

containers=("preprocess-cifar10:latest" "cifar10-model-save:latest")
for container in "${containers[@]}"
do
  docker pull $CONTAINER_REGISTRY"/"$container
done