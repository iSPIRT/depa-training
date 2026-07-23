#!/bin/bash

set -euo pipefail

for required_var in \
  AZURE_RESOURCE_GROUP \
  AZURE_LOCATION \
  AZURE_STORAGE_ACCOUNT_NAME \
  AZURE_BANK_1_CONTAINER_NAME \
  AZURE_BANK_2_CONTAINER_NAME \
  AZURE_BANK_3_CONTAINER_NAME \
  AZURE_OUTPUT_CONTAINER_NAME
do
  if [[ -z "${!required_var:-}" ]]; then
    echo "Environment variable $required_var not defined" >&2
    exit 1
  fi
done

echo "Checking if resource group $AZURE_RESOURCE_GROUP exists..."
RG_EXISTS=$(az group exists --name $AZURE_RESOURCE_GROUP)

if [ "$RG_EXISTS" == "false" ]; then
  echo "Resource group $AZURE_RESOURCE_GROUP does not exist. Creating it now..."
  # Create the resource group
  az group create --name $AZURE_RESOURCE_GROUP --location $AZURE_LOCATION
else
  echo "Resource group $AZURE_RESOURCE_GROUP already exists. Skipping creation."
fi

echo "Check if storage account $AZURE_STORAGE_ACCOUNT_NAME exists..."
STORAGE_ACCOUNT_EXISTS=$(az storage account check-name --name $AZURE_STORAGE_ACCOUNT_NAME --query "nameAvailable" --output tsv)

if [ "$STORAGE_ACCOUNT_EXISTS" == "true" ]; then
  echo "Storage account $AZURE_STORAGE_ACCOUNT_NAME does not exist. Creating it now..."
  az storage account create --resource-group "$AZURE_RESOURCE_GROUP" \
    --name "$AZURE_STORAGE_ACCOUNT_NAME" --location "$AZURE_LOCATION"
else
  echo "Storage account $AZURE_STORAGE_ACCOUNT_NAME already exists. Skipping creation."
fi

# Get the storage account key
ACCOUNT_KEY=$(az storage account keys list --resource-group $AZURE_RESOURCE_GROUP --account-name $AZURE_STORAGE_ACCOUNT_NAME --query "[0].value" --output tsv)


# Check if the BANK_1 container exists
CONTAINER_EXISTS=$(az storage container exists --name $AZURE_BANK_1_CONTAINER_NAME --account-name $AZURE_STORAGE_ACCOUNT_NAME --account-key $ACCOUNT_KEY --query "exists" --output tsv)

if [ "$CONTAINER_EXISTS" == "false" ]; then
    echo "Container $AZURE_BANK_1_CONTAINER_NAME does not exist. Creating it now..."
    az storage container create --resource-group $AZURE_RESOURCE_GROUP --account-name $AZURE_STORAGE_ACCOUNT_NAME --name $AZURE_BANK_1_CONTAINER_NAME --account-key $ACCOUNT_KEY
fi

# Check if the BANK_2 container exists
CONTAINER_EXISTS=$(az storage container exists --name $AZURE_BANK_2_CONTAINER_NAME --account-name $AZURE_STORAGE_ACCOUNT_NAME --account-key $ACCOUNT_KEY --query "exists" --output tsv)

if [ "$CONTAINER_EXISTS" == "false" ]; then
    echo "Container $AZURE_BANK_2_CONTAINER_NAME does not exist. Creating it now..."
    az storage container create --resource-group $AZURE_RESOURCE_GROUP --account-name $AZURE_STORAGE_ACCOUNT_NAME --name $AZURE_BANK_2_CONTAINER_NAME --account-key $ACCOUNT_KEY
fi

# Check if the BANK_3 container exists
CONTAINER_EXISTS=$(az storage container exists --name $AZURE_BANK_3_CONTAINER_NAME --account-name $AZURE_STORAGE_ACCOUNT_NAME --account-key $ACCOUNT_KEY --query "exists" --output tsv)

if [ "$CONTAINER_EXISTS" == "false" ]; then
    echo "Container $AZURE_BANK_3_CONTAINER_NAME does not exist. Creating it now..."
    az storage container create --resource-group $AZURE_RESOURCE_GROUP --account-name $AZURE_STORAGE_ACCOUNT_NAME --name $AZURE_BANK_3_CONTAINER_NAME --account-key $ACCOUNT_KEY
fi

# Check if the OUTPUT container exists
CONTAINER_EXISTS=$(az storage container exists --name $AZURE_OUTPUT_CONTAINER_NAME --account-name $AZURE_STORAGE_ACCOUNT_NAME --account-key $ACCOUNT_KEY --query "exists" --output tsv)

if [ "$CONTAINER_EXISTS" == "false" ]; then
  echo "Container $AZURE_OUTPUT_CONTAINER_NAME does not exist. Creating it now..."
  az storage container create --resource-group $AZURE_RESOURCE_GROUP --account-name $AZURE_STORAGE_ACCOUNT_NAME --name $AZURE_OUTPUT_CONTAINER_NAME --account-key $ACCOUNT_KEY
fi
