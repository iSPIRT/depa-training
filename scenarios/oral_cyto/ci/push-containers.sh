#!/bin/bash
docker tag preprocess-oral-cyto-a:latest $CONTAINER_REGISTRY/preprocess-oral-cyto-a:latest
docker push $CONTAINER_REGISTRY/preprocess-oral-cyto-a:latest
docker tag preprocess-oral-cyto-b:latest $CONTAINER_REGISTRY/preprocess-oral-cyto-b:latest
docker push $CONTAINER_REGISTRY/preprocess-oral-cyto-b:latest
docker tag oral-cyto-model-save:latest $CONTAINER_REGISTRY/oral-cyto-model-save:latest
docker push $CONTAINER_REGISTRY/oral-cyto-model-save:latest
