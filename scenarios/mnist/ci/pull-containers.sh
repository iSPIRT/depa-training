#!/bin/bash

containers=("preprocess-mnist:latest" "mnist-model-save:latest")
for container in "${containers[@]}"
do
  docker pull $CONTAINER_REGISTRY"/"$container
done