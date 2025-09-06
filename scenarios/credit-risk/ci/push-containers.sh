containers=("preprocess-bank-a:latest" "preprocess-bank-b:latest" "preprocess-bureau:latest" "preprocess-fintech:latest")
for container in "${containers[@]}"
do
  docker tag $container $CONTAINER_REGISTRY"/"$container
  docker push $CONTAINER_REGISTRY"/"$container
done
