#!/bin/bash

az group create \
    --location $AZURE_LOCATION\\
    --name $AZURE_RESOURCE_GROUP

az storage account create \
    --resource-group $AZURE_RESOURCE_GROUP \
    --name $AZURE_STORAGE_ACCOUNT_NAME

az storage container create \
    --resource-group $AZURE_RESOURCE_GROUP \
    --account-name $AZURE_STORAGE_ACCOUNT_NAME \
    --name $AZURE_ICMR_CONTAINER_NAME 

az storage container create \
    --resource-group $AZURE_RESOURCE_GROUP \
    --account-name $AZURE_STORAGE_ACCOUNT_NAME \
    --name $AZURE_COWIN_CONTAINER_NAME 

az storage container create \
    --resource-group $AZURE_RESOURCE_GROUP \
    --account-name $AZURE_STORAGE_ACCOUNT_NAME \
    --name $AZURE_INDEX_CONTAINER_NAME 

az storage container create \
    --resource-group $AZURE_RESOURCE_GROUP \
    --account-name $AZURE_STORAGE_ACCOUNT_NAME \
    --name $AZURE_MODEL_CONTAINER_NAME 

az storage container create \
    --resource-group $AZURE_RESOURCE_GROUP \
    --account-name $AZURE_STORAGE_ACCOUNT_NAME \
    --name $AZURE_OUTPUT_CONTAINER_NAME 
