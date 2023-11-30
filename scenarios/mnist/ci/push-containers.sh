containers=("preprocess-mnist_1:latest" "preprocess-mnist_2:latest" "preprocess-mnist_3:latest" "ccr-model-save:latest")
for container in "${containers[@]}"
do
  docker tag $container $CONTAINER_REGISTRY"/"$container
  docker push $CONTAINER_REGISTRY"/"$container
done
