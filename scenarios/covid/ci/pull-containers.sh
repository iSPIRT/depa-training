#!/bin/bash

containers=("preprocess-icmr:latest" "preprocess-cowin:latest" "preprocess-index:latest" "covid-model-save:latest")
for container in "${containers[@]}"
do
  docker pull $CONTAINER_REGISTRY"/"$container
done