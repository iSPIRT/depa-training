#!/bin/bash

containers=("depa-training:latest" "depa-training-encfs:latest")
for container in "${containers[@]}"
do
  docker tag $container $CONTAINER_REGISTRY"/"$container
  docker push $CONTAINER_REGISTRY"/"$container
done
