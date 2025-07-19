#!/bin/bash

ACCOUNT_KEY=$(az storage account keys list --account-name $AZURE_STORAGE_ACCOUNT_NAME --only-show-errors | jq -r .[0].value)

az storage blob upload \
  --account-name $AZURE_STORAGE_ACCOUNT_NAME \
  --container $AZURE_BRATS_A_CONTAINER_NAME \
  --file brats_a.img \
  --name data.img \
  --type page \
  --overwrite \
  --account-key $ACCOUNT_KEY

az storage blob upload \
  --account-name $AZURE_STORAGE_ACCOUNT_NAME \
  --container $AZURE_BRATS_B_CONTAINER_NAME \
  --file brats_b.img \
  --name data.img \
  --type page \
  --overwrite \
  --account-key $ACCOUNT_KEY

az storage blob upload \
  --account-name $AZURE_STORAGE_ACCOUNT_NAME \
  --container $AZURE_BRATS_C_CONTAINER_NAME \
  --file brats_c.img \
  --name data.img \
  --type page \
  --overwrite \
  --account-key $ACCOUNT_KEY

az storage blob upload \
  --account-name $AZURE_STORAGE_ACCOUNT_NAME \
  --container $AZURE_BRATS_D_CONTAINER_NAME \
  --file brats_d.img \
  --name data.img \
  --type page \
  --overwrite \
  --account-key $ACCOUNT_KEY

az storage blob upload \
  --account-name $AZURE_STORAGE_ACCOUNT_NAME \
  --container $AZURE_MODEL_CONTAINER_NAME \
  --file model.img \
  --name data.img \
  --type page \
  --overwrite \
  --account-key $ACCOUNT_KEY

az storage blob upload \
  --account-name $AZURE_STORAGE_ACCOUNT_NAME \
  --container $AZURE_OUTPUT_CONTAINER_NAME \
  --file output.img \
  --name data.img \
  --type page \
  --overwrite \
  --account-key $ACCOUNT_KEY
