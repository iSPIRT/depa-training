#!/bin/bash
docker tag preprocess-bank-1:latest $CONTAINER_REGISTRY/preprocess-bank-1:latest
docker push $CONTAINER_REGISTRY/preprocess-bank-1:latest
docker tag preprocess-bank-2:latest $CONTAINER_REGISTRY/preprocess-bank-2:latest
docker push $CONTAINER_REGISTRY/preprocess-bank-2:latest
docker tag preprocess-bank-3:latest $CONTAINER_REGISTRY/preprocess-bank-3:latest
docker push $CONTAINER_REGISTRY/preprocess-bank-3:latest
