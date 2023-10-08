#!/bin/bash

containers=("ccr-model-save:latest" "depa-training:latest" "depa-training-encfs:latest")
for container in "${containers[@]}"
do
  docker pull $CONTAINER_REGISTRY"/"$container
done