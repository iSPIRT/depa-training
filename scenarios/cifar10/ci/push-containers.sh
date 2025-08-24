containers=("cifar10-model-save:latest" "preprocess-cifar10:latest")
for container in "${containers[@]}"
do
  docker tag $container $CONTAINER_REGISTRY"/"$container
  docker push $CONTAINER_REGISTRY"/"$container
done
