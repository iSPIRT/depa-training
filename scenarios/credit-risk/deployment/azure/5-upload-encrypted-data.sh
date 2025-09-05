#!/bin/bash

export DATA_DIR=$REPO_ROOT/scenarios/$SCENARIO/data
export MODEL_DIR=$REPO_ROOT/scenarios/$SCENARIO/modeller

ACCOUNT_KEY=$(az storage account keys list --account-name $AZURE_STORAGE_ACCOUNT_NAME --only-show-errors | jq -r .[0].value)

az storage blob upload \
  --account-name $AZURE_STORAGE_ACCOUNT_NAME \
  --container $AZURE_BANK_A_CONTAINER_NAME \
  --file $DATA_DIR/bank_a.img \
  --name data.img \
  --type page \
  --overwrite \
  --account-key $ACCOUNT_KEY

az storage blob upload \
  --account-name $AZURE_STORAGE_ACCOUNT_NAME \
  --container $AZURE_BANK_B_CONTAINER_NAME \
  --file $DATA_DIR/bank_b.img \
  --name data.img \
  --type page \
  --overwrite \
  --account-key $ACCOUNT_KEY

az storage blob upload \
  --account-name $AZURE_STORAGE_ACCOUNT_NAME \
  --container $AZURE_BUREAU_CONTAINER_NAME \
  --file $DATA_DIR/bureau.img \
  --name data.img \
  --type page \
  --overwrite \
  --account-key $ACCOUNT_KEY

az storage blob upload \
  --account-name $AZURE_STORAGE_ACCOUNT_NAME \
  --container $AZURE_FINTECH_CONTAINER_NAME \
  --file $DATA_DIR/fintech.img \
  --name data.img \
  --type page \
  --overwrite \
  --account-key $ACCOUNT_KEY

az storage blob upload \
  --account-name $AZURE_STORAGE_ACCOUNT_NAME \
  --container $AZURE_OUTPUT_CONTAINER_NAME \
  --file $MODEL_DIR/output.img \
  --name data.img \
  --type page \
  --overwrite \
  --account-key $ACCOUNT_KEY
