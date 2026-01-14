#!/bin/bash
#
echo "Checking if resource group $AZURE_RESOURCE_GROUP exists..."
RG_EXISTS=$(az group exists --name $AZURE_RESOURCE_GROUP)

if [ "$RG_EXISTS" == "false" ]; then
  echo "Resource group $AZURE_RESOURCE_GROUP does not exist. Creating it now..."
  # Create the resource group
  az group create --name $AZURE_RESOURCE_GROUP --location $AZURE_LOCATION
else
  echo "Resource group $AZURE_RESOURCE_GROUP already exists. Skipping creation."
fi

echo "Check if storage account $STORAGE_ACCOUNT_NAME exists..."
STORAGE_ACCOUNT_EXISTS=$(az storage account check-name --name $AZURE_STORAGE_ACCOUNT_NAME --query "nameAvailable" --output tsv)

if [ "$STORAGE_ACCOUNT_EXISTS" == "true" ]; then
  echo "Storage account $AZURE_STORAGE_ACCOUNT_NAME does not exist. Creating it now..."
  az storage account create  --resource-group $AZURE_RESOURCE_GROUP  --name $AZURE_STORAGE_ACCOUNT_NAME
else
  # Check if the storage account exists in the resource group
  STORAGE_ACCOUNT_EXIST_IN_RG=$(az storage account show --name $AZURE_STORAGE_ACCOUNT_NAME --resource-group $AZURE_RESOURCE_GROUP --query "name" -o tsv 2>/dev/null)
  if [ -z "$STORAGE_ACCOUNT_EXIST_IN_RG" ]; then
    echo "Storage account $AZURE_STORAGE_ACCOUNT_NAME is already reserved and does not exist in the $AZURE_RESOURCE_GROUP resource group. Please select a different name."
    exit 1
  fi
  echo "Storage account $AZURE_STORAGE_ACCOUNT_NAME already exists in the $AZURE_RESOURCE_GROUP resource group. Skipping creation."
fi

# Get the storage account key
ACCOUNT_KEY=$(az storage account keys list --resource-group $AZURE_RESOURCE_GROUP --account-name $AZURE_STORAGE_ACCOUNT_NAME --query "[0].value" --output tsv)


# Check if the ICMR container exists
CONTAINER_EXISTS=$(az storage container exists --name $AZURE_ICMR_CONTAINER_NAME --account-name $AZURE_STORAGE_ACCOUNT_NAME --account-key $ACCOUNT_KEY --query "exists" --output tsv)

if [ "$CONTAINER_EXISTS" == "false" ]; then
  echo "Container $AZURE_ICMR_CONTAINER_NAME does not exist. Creating it now..."
  az storage container create --resource-group $AZURE_RESOURCE_GROUP --account-name $AZURE_STORAGE_ACCOUNT_NAME --name $AZURE_ICMR_CONTAINER_NAME --account-key $ACCOUNT_KEY
fi

# Check if the COWIN container exists
CONTAINER_EXISTS=$(az storage container exists --name $AZURE_COWIN_CONTAINER_NAME --account-name $AZURE_STORAGE_ACCOUNT_NAME --account-key $ACCOUNT_KEY --query "exists" --output tsv)

if [ "$CONTAINER_EXISTS" == "false" ]; then
  echo "Container $AZURE_COWIN_CONTAINER_NAME does not exist. Creating it now..."
  az storage container create --resource-group $AZURE_RESOURCE_GROUP --account-name $AZURE_STORAGE_ACCOUNT_NAME --name $AZURE_COWIN_CONTAINER_NAME --account-key $ACCOUNT_KEY
fi

# Check if the INDEX container exists
CONTAINER_EXISTS=$(az storage container exists --name $AZURE_INDEX_CONTAINER_NAME --account-name $AZURE_STORAGE_ACCOUNT_NAME --account-key $ACCOUNT_KEY --query "exists" --output tsv)

if [ "$CONTAINER_EXISTS" == "false" ]; then
  echo "Container $AZURE_INDEX_CONTAINER_NAME does not exist. Creating it now..."
  az storage container create --resource-group $AZURE_RESOURCE_GROUP --account-name $AZURE_STORAGE_ACCOUNT_NAME --name $AZURE_INDEX_CONTAINER_NAME --account-key $ACCOUNT_KEY
fi

# Check if the MODEL container exists
CONTAINER_EXISTS=$(az storage container exists --name $AZURE_MODEL_CONTAINER_NAME --account-name $AZURE_STORAGE_ACCOUNT_NAME --account-key $ACCOUNT_KEY --query "exists" --output tsv)

if [ "$CONTAINER_EXISTS" == "false" ]; then
  echo "Container $AZURE_MODEL_CONTAINER_NAME does not exist. Creating it now..."
  az storage container create --resource-group $AZURE_RESOURCE_GROUP --account-name $AZURE_STORAGE_ACCOUNT_NAME --name $AZURE_MODEL_CONTAINER_NAME --account-key $ACCOUNT_KEY
fi

# Check if the OUTPUT container exists
CONTAINER_EXISTS=$(az storage container exists --name $AZURE_OUTPUT_CONTAINER_NAME --account-name $AZURE_STORAGE_ACCOUNT_NAME --account-key $ACCOUNT_KEY --query "exists" --output tsv)

if [ "$CONTAINER_EXISTS" == "false" ]; then
  echo "Container $AZURE_OUTPUT_CONTAINER_NAME does not exist. Creating it now..."
  az storage container create --resource-group $AZURE_RESOURCE_GROUP --account-name $AZURE_STORAGE_ACCOUNT_NAME --name $AZURE_OUTPUT_CONTAINER_NAME --account-key $ACCOUNT_KEY
fi
