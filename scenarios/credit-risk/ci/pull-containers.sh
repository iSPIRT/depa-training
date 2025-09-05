#!/bin/bash

containers=("preprocess-bank-a:latest" "preprocess-bank-b:latest" "preprocess-bureau:latest" "preprocess-fintech:latest")
for container in "${containers[@]}"
do
  docker pull $CONTAINER_REGISTRY"/"$container
done