#!/bin/bash

set -e 

if [[ "$AZURE_KEYVAULT_ENDPOINT" == *".vault.azure.net" ]]; then 
    AZURE_AKV_RESOURCE_NAME=`echo $AZURE_KEYVAULT_ENDPOINT | awk '{split($0,a,"."); print a[1]}'`
    # Check if the Key Vault already exists
    echo "Checking if Key Vault $AZURE_AKV_RESOURCE_NAME exists..."
    NAME_AVAILABLE=$(az rest --method post \
        --uri "https://management.azure.com/subscriptions/$AZURE_SUBSCRIPTION_ID/providers/Microsoft.KeyVault/checkNameAvailability?api-version=2019-09-01" \
        --headers "Content-Type=application/json" \
        --body "{\"name\": \"$AZURE_AKV_RESOURCE_NAME\", \"type\": \"Microsoft.KeyVault/vaults\"}" | jq -r '.nameAvailable')
    if [ "$NAME_AVAILABLE" == true ]; then
        echo "Key Vault $AZURE_AKV_RESOURCE_NAME does not exist. Creating it now..."
        echo CREATING $AZURE_KEYVAULT_ENDPOINT in resource group $AZURE_RESOURCE_GROUP
        # Create Azure key vault with RBAC authorization
        az keyvault create --name $AZURE_AKV_RESOURCE_NAME --resource-group $AZURE_RESOURCE_GROUP --sku "Premium" --enable-rbac-authorization
        # Assign RBAC roles to the resource owner so they can import keys
        AKV_SCOPE=`az keyvault show --name $AZURE_AKV_RESOURCE_NAME --query id --output tsv`    
        az role assignment create --role "Key Vault Crypto Officer" --assignee `az account show --query user.name --output tsv` --scope $AKV_SCOPE
        az role assignment create --role "Key Vault Crypto User" --assignee `az account show --query user.name --output tsv` --scope $AKV_SCOPE
    else
        # Name is not available — check if the vault exists in *this* subscription
        if az keyvault show --name $AZURE_AKV_RESOURCE_NAME --resource-group $AZURE_RESOURCE_GROUP >/dev/null 2>&1; then
            echo "Key Vault $AZURE_AKV_RESOURCE_NAME already exists in your subscription. Skipping creation."
        else
            echo "Key Vault name '$AZURE_AKV_RESOURCE_NAME' is already reserved."
            echo "If you previously owned this Key Vault, it may be soft-deleted."
            echo "Please purge the Key Vault before retrying."
            exit 1
        fi
    fi
else
    echo "Automated creation of key vaults is supported only for vaults"
fi
