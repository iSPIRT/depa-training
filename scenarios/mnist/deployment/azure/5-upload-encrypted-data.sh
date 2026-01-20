#!/bin/bash

DATADIR=$REPO_ROOT/scenarios/$SCENARIO/data
MODELDIR=$REPO_ROOT/scenarios/$SCENARIO/modeller

ACCOUNT_KEY=$(az storage account keys list --account-name $AZURE_STORAGE_ACCOUNT_NAME --only-show-errors | jq -r .[0].value)

az storage blob upload \
  --account-name $AZURE_STORAGE_ACCOUNT_NAME \
  --container $AZURE_MNIST_CONTAINER_NAME \
  --file $DATADIR/mnist.img \
  --name data.img \
  --type page \
  --overwrite \
  --account-key $ACCOUNT_KEY

az storage blob upload \
  --account-name $AZURE_STORAGE_ACCOUNT_NAME \
  --container $AZURE_MODEL_CONTAINER_NAME \
  --file $MODELDIR/model.img \
  --name data.img \
  --type page \
  --overwrite \
  --account-key $ACCOUNT_KEY

az storage blob upload \
  --account-name $AZURE_STORAGE_ACCOUNT_NAME \
  --container $AZURE_OUTPUT_CONTAINER_NAME \
  --file $MODELDIR/output.img \
  --name data.img \
  --type page \
  --overwrite \
  --account-key $ACCOUNT_KEY
