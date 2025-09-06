#!/bin/bash

# Only to be run when creating a new ACR

# Ensure required env vars are set
if [[ -z "$CONTAINER_REGISTRY" || -z "$AZURE_RESOURCE_GROUP" || -z "$AZURE_LOCATION" ]]; then
  echo "ERROR: CONTAINER_REGISTRY, AZURE_RESOURCE_GROUP, and AZURE_LOCATION environment variables must be set."
  exit 1
fi

echo "Checking if ACR '$CONTAINER_REGISTRY' exists in resource group '$AZURE_RESOURCE_GROUP'..."

# Check if ACR exists
ACR_EXISTS=$(az acr show --name "$CONTAINER_REGISTRY" --resource-group "$AZURE_RESOURCE_GROUP" --query "name" -o tsv 2>/dev/null)

if [[ -n "$ACR_EXISTS" ]]; then
  echo "✅ ACR '$CONTAINER_REGISTRY' already exists."
else
  echo "⏳ ACR '$CONTAINER_REGISTRY' does not exist. Creating..."

  # Create ACR with premium SKU and admin enabled
  az acr create \
    --name "$CONTAINER_REGISTRY" \
    --resource-group "$AZURE_RESOURCE_GROUP" \
    --location "$AZURE_LOCATION" \
    --sku Premium \
    --admin-enabled true \
    --output table

  # Enable anonymous pull
  az acr update --name "$CONTAINER_REGISTRY" --anonymous-pull-enabled true

  if [[ $? -eq 0 ]]; then
    echo "✅ ACR '$CONTAINER_REGISTRY' created successfully."
  else
    echo "❌ Failed to create ACR."
    exit 1
  fi
fi

# Login to the ACR
az acr login --name "$CONTAINER_REGISTRY"