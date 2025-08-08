containers=("brats-model-save:latest" "preprocess-brats-a:latest" "preprocess-brats-b:latest" "preprocess-brats-c:latest" "preprocess-brats-d:latest")
for container in "${containers[@]}"
do
  docker tag $container $CONTAINER_REGISTRY"/"$container
  docker push $CONTAINER_REGISTRY"/"$container
done
