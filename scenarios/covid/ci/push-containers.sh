containers=("preprocess-icmr:latest" "preprocess-cowin:latest" "preprocess-index:latest" "covid-model-save:latest")
for container in "${containers[@]}"
do
  docker tag $container $CONTAINER_REGISTRY"/"$container
  docker push $CONTAINER_REGISTRY"/"$container
done
