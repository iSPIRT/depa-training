#!/bin/bash

# Only to be run when creating a new ACR

# Ensure required env vars are set
if [[ -z "$CONTAINER_REGISTRY" || -z "$AZURE_RESOURCE_GROUP" || -z "$AZURE_LOCATION" ]]; then
  echo "ERROR: CONTAINER_REGISTRY, AZURE_RESOURCE_GROUP, and AZURE_LOCATION environment variables must be set."
  exit 1
fi

# Remove .azurecr.io suffix if present (ACR names should not include the domain)
CONTAINER_REGISTRY="${CONTAINER_REGISTRY%.azurecr.io}"

# Validate ACR name: alphanumeric only, 5-50 characters
if [[ ! "$CONTAINER_REGISTRY" =~ ^[a-zA-Z0-9]{5,50}$ ]]; then
  echo "ERROR: Invalid ACR name '$CONTAINER_REGISTRY'. ACR names must be 5-50 alphanumeric characters only."
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
    echo "⏳ Waiting for DNS propagation before login..."
    sleep 10
  else
    echo "❌ Failed to create ACR."
    exit 1
  fi
fi

# Login to the ACR with retry mechanism (DNS propagation may take time)
echo "Logging in to ACR '$CONTAINER_REGISTRY'..."
MAX_RETRIES=15
RETRY_DELAY=5
RETRY_COUNT=0

while [[ $RETRY_COUNT -lt $MAX_RETRIES ]]; do
  if az acr login --name "$CONTAINER_REGISTRY" 2>/dev/null; then
    echo "✅ Successfully logged in to ACR '$CONTAINER_REGISTRY'."
    exit 0
  fi
  
  RETRY_COUNT=$((RETRY_COUNT + 1))
  if [[ $RETRY_COUNT -lt $MAX_RETRIES ]]; then
    echo "⏳ Login attempt $RETRY_COUNT/$MAX_RETRIES failed. Retrying in ${RETRY_DELAY}s..."
    sleep $RETRY_DELAY
    RETRY_DELAY=$((RETRY_DELAY + 5))  # Exponential backoff
  fi
done

echo "❌ Failed to login to ACR after $MAX_RETRIES attempts. DNS may still be propagating."
echo "You can manually login later with: az acr login --name $CONTAINER_REGISTRY"
exit 1