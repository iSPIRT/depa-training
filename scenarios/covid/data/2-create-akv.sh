#!/bin/bash

set -e 

  echo CREATING $AZURE_KEYVAULT_ENDPOINT in resouce group $AZURE_RESOURCE_GROUP
  
if [[ "$AZURE_KEYVAULT_ENDPOINT" == *".vault.azure.net" ]]; then 
    AZURE_AKV_RESOURCE_NAME=`echo $AZURE_KEYVAULT_ENDPOINT | awk '{split($0,a,"."); print a[1]}'`
    # Check if the Key Vault already exists
    echo "Checking if Key Vault $KEY_VAULT_NAME exists..."
    KEY_VAULT_EXISTS=$(az keyvault list --resource-group $AZURE_RESOURCE_GROUP --query "[?name=='$AZURE_AKV_RESOURCE_NAME'].name" --output tsv)
    if [ -z "$KEY_VAULT_EXISTS" ]; then
        echo "Key Vault $KEY_VAULT_NAME does not exist. Creating it now..."
        # Create Azure key vault with RBAC authorization
        az keyvault create --name $AZURE_AKV_RESOURCE_NAME --resource-group $AZURE_RESOURCE_GROUP --sku "Premium" --enable-rbac-authorization
        # Assign RBAC roles to the resource owner so they can import keys
        AKV_SCOPE=`az keyvault show --name $AZURE_AKV_RESOURCE_NAME --query id --output tsv`    
        az role assignment create --role "Key Vault Crypto Officer" --assignee `az account show --query user.name --output tsv` --scope $AKV_SCOPE
        az role assignment create --role "Key Vault Crypto User" --assignee `az account show --query user.name --output tsv` --scope $AKV_SCOPE
    else
	echo "Key Vault $AZURE_KEYVAULT_ENDPOINT exists"
    fi
else
    echo "Automated creation of key vaults is supported only for vaults"
fi
