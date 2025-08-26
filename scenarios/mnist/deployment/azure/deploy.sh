#!/bin/bash

set -e 

while getopts ":c:p:" options; do
    case $options in 
        c)contract=$OPTARG;;
        p)pipelineConfiguration=$OPTARG;;
    esac
done

if [[ -z "${contract}" ]]; then
  echo "No contract specified"
  exit 1
fi

if [[ -z "${pipelineConfiguration}" ]]; then
  echo "No pipeline configuration specified"
  exit 1
fi

if [[ -z "${AZURE_KEYVAULT_ENDPOINT}" ]]; then
  echo "Environment variable AZURE_KEYVAULT_ENDPOINT not defined"
fi

echo Obtaining contract service parameters...

CONTRACT_SERVICE_URL=${CONTRACT_SERVICE_URL:-"https://localhost:8000"}
export CONTRACT_SERVICE_PARAMETERS=$(curl -k -f $CONTRACT_SERVICE_URL/parameters | base64 --wrap=0)

echo Computing CCE policy...
envsubst < ../../policy/policy-in-template.json > /tmp/policy-in.json
export CCE_POLICY=$(az confcom acipolicygen -i /tmp/policy-in.json --debug-mode)
export CCE_POLICY_HASH=$(go run $TOOLS_HOME/securitypolicydigest/main.go -p $CCE_POLICY)
echo "Training container policy hash $CCE_POLICY_HASH"

export CONTRACTS=$contract
export PIPELINE_CONFIGURATION=`cat $pipelineConfiguration | base64 --wrap=0`

function generate_encrypted_filesystem_information() {
  end=`date -u -d "60 minutes" '+%Y-%m-%dT%H:%MZ'`
  MNIST_SAS_TOKEN=$(az storage blob generate-sas --account-name $AZURE_STORAGE_ACCOUNT_NAME --container-name $AZURE_MNIST_CONTAINER_NAME --permissions r --name data.img --expiry $end --only-show-errors) 
  export MNIST_SAS_TOKEN="$(echo -n $MNIST_SAS_TOKEN | tr -d \")"
  export MNIST_SAS_TOKEN="?$MNIST_SAS_TOKEN"

  MODEL_SAS_TOKEN=$(az storage blob generate-sas --account-name $AZURE_STORAGE_ACCOUNT_NAME --container-name $AZURE_MODEL_CONTAINER_NAME --permissions r --name data.img --expiry $end --only-show-errors) 
  export MODEL_SAS_TOKEN=$(echo $MODEL_SAS_TOKEN | tr -d \")
  export MODEL_SAS_TOKEN="?$MODEL_SAS_TOKEN"

  OUTPUT_SAS_TOKEN=$(az storage blob generate-sas --account-name $AZURE_STORAGE_ACCOUNT_NAME --container-name $AZURE_OUTPUT_CONTAINER_NAME --permissions rw --name data.img --expiry $end --only-show-errors) 
  export OUTPUT_SAS_TOKEN=$(echo $OUTPUT_SAS_TOKEN | tr -d \")
  export OUTPUT_SAS_TOKEN="?$OUTPUT_SAS_TOKEN"

  # Obtain the token based on the AKV resource endpoint subdomain
  if [[ "$AZURE_KEYVAULT_ENDPOINT" == *".vault.azure.net" ]]; then
      export BEARER_TOKEN=$(az account get-access-token --resource https://vault.azure.net | jq -r .accessToken)
      echo "Importing keys to AKV key vaults can be only of type RSA-HSM"
      export AZURE_AKV_KEY_TYPE="RSA-HSM"
  elif [[ "$AZURE_KEYVAULT_ENDPOINT" == *".managedhsm.azure.net" ]]; then
      export BEARER_TOKEN=$(az account get-access-token --resource https://managedhsm.azure.net | jq -r .accessToken)
      export AZURE_AKV_KEY_TYPE="oct-HSM"
  fi

  TMP=$(jq . encrypted-filesystem-config-template.json)
  TMP=`echo $TMP | \
    jq '.azure_filesystems[0].azure_url = "https://" + env.AZURE_STORAGE_ACCOUNT_NAME + ".blob.core.windows.net/" + env.AZURE_MNIST_CONTAINER_NAME + "/data.img" + env.MNIST_SAS_TOKEN' | \
    jq '.azure_filesystems[0].mount_point = "/mnt/remote/mnist"' | \
    jq '.azure_filesystems[0].key.kid = "MNISTFilesystemEncryptionKey"' | \
    jq '.azure_filesystems[0].key.kty = env.AZURE_AKV_KEY_TYPE' | \
    jq '.azure_filesystems[0].key.akv.endpoint = env.AZURE_KEYVAULT_ENDPOINT' | \
    jq '.azure_filesystems[0].key.akv.bearer_token = env.BEARER_TOKEN' | \
    jq '.azure_filesystems[0].key_derivation.label = "MNISTFilesystemEncryptionKey"' | \
    jq '.azure_filesystems[0].key_derivation.salt = "9b53cddbe5b78a0b912a8f05f341bcd4dd839ea85d26a08efaef13e696d999f4"'`

  TMP=`echo $TMP | \
    jq '.azure_filesystems[1].azure_url = "https://" + env.AZURE_STORAGE_ACCOUNT_NAME + ".blob.core.windows.net/" + env.AZURE_MODEL_CONTAINER_NAME + "/data.img" + env.MODEL_SAS_TOKEN' | \
    jq '.azure_filesystems[1].mount_point = "/mnt/remote/model"' | \
    jq '.azure_filesystems[1].key.kid = "ModelFilesystemEncryptionKey"' | \
    jq '.azure_filesystems[1].key.kty = env.AZURE_AKV_KEY_TYPE' | \
    jq '.azure_filesystems[1].key.akv.endpoint = env.AZURE_KEYVAULT_ENDPOINT' | \
    jq '.azure_filesystems[1].key.akv.bearer_token = env.BEARER_TOKEN' | \
    jq '.azure_filesystems[1].key_derivation.label = "ModelFilesystemEncryptionKey"' | \
    jq '.azure_filesystems[1].key_derivation.salt = "9b53cddbe5b78a0b912a8f05f341bcd4dd839ea85d26a08efaef13e696d999f4"'`

  TMP=`echo $TMP | \
    jq '.azure_filesystems[2].azure_url = "https://" + env.AZURE_STORAGE_ACCOUNT_NAME + ".blob.core.windows.net/" + env.AZURE_OUTPUT_CONTAINER_NAME + "/data.img" + env.OUTPUT_SAS_TOKEN' | \
    jq '.azure_filesystems[2].mount_point = "/mnt/remote/output"' | \
    jq '.azure_filesystems[2].key.kid = "OutputFilesystemEncryptionKey"' | \
    jq '.azure_filesystems[2].key.kty = env.AZURE_AKV_KEY_TYPE' | \
    jq '.azure_filesystems[2].key.akv.endpoint = env.AZURE_KEYVAULT_ENDPOINT' | \
    jq '.azure_filesystems[2].key.akv.bearer_token = env.BEARER_TOKEN' | \
    jq '.azure_filesystems[2].key_derivation.label = "OutputFilesystemEncryptionKey"' | \
    jq '.azure_filesystems[2].key_derivation.salt = "9b53cddbe5b78a0b912a8f05f341bcd4dd839ea85d26a08efaef13e696d999f4"'`

  ENCRYPTED_FILESYSTEM_INFORMATION=`echo $TMP | base64 --wrap=0`
}

echo Generating encrypted file system information...
generate_encrypted_filesystem_information
echo $ENCRYPTED_FILESYSTEM_INFORMATION > /tmp/encrypted-filesystem-config.json
export ENCRYPTED_FILESYSTEM_INFORMATION

echo Generating parameters for ACI deployment...
TMP=$(jq '.containerRegistry.value = env.CONTAINER_REGISTRY' aci-parameters-template.json)
TMP=`echo $TMP | jq '.ccePolicy.value = env.CCE_POLICY'`
TMP=`echo $TMP | jq '.EncfsSideCarArgs.value = env.ENCRYPTED_FILESYSTEM_INFORMATION'`
TMP=`echo $TMP | jq '.ContractService.value = env.CONTRACT_SERVICE_URL'`
TMP=`echo $TMP | jq '.ContractServiceParameters.value = env.CONTRACT_SERVICE_PARAMETERS'`
TMP=`echo $TMP | jq '.Contracts.value = env.CONTRACTS'`
TMP=`echo $TMP | jq '.PipelineConfiguration.value = env.PIPELINE_CONFIGURATION'`
echo $TMP > /tmp/aci-parameters.json

echo Deploying training clean room...

echo "Checking if resource group $AZURE_RESOURCE_GROUP exists..."
RG_EXISTS=$(az group exists --name $AZURE_RESOURCE_GROUP)

if [ "$RG_EXISTS" == "false" ]; then
  echo "Resource group $AZURE_RESOURCE_GROUP does not exist. Creating it now..."
  # Create the resource group
  az group create --name $AZURE_RESOURCE_GROUP --location $AZURE_LOCATION
else
  echo "Resource group $AZURE_RESOURCE_GROUP already exists. Skipping creation."
fi

az deployment group create \
  --resource-group $AZURE_RESOURCE_GROUP \
  --template-file arm-template.json \
  --parameters @/tmp/aci-parameters.json

echo Deployment complete. 