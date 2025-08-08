containers=("preprocess-medqa:latest" "preprocess-chatdoctor:latest" "preprocess-medquad" "ccr-model-save:latest")
for container in "${containers[@]}"
do
  docker tag $container $CONTAINER_REGISTRY"/"$container
  docker push $CONTAINER_REGISTRY"/"$container
done