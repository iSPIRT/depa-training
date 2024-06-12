containers=("depa-mnist-save-model:latest" "depa-mnist-preprocess:latest")
for container in "${containers[@]}"
do
  docker tag $container $CONTAINER_REGISTRY"/"$container
  docker push $CONTAINER_REGISTRY"/"$container
done
