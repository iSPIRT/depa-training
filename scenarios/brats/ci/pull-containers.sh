#!/bin/bash

containers=("preprocess-brats-a:latest" "preprocess-brats-b:latest" "preprocess-brats-c:latest" "preprocess-brats-d:latest" "brats-model-save:latest")
for container in "${containers[@]}"
do
  docker pull $CONTAINER_REGISTRY"/"$container
done