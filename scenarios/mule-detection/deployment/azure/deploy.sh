#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/validate-contract-service-url.sh"

for required_var in \
  AZURE_KEYVAULT_ENDPOINT \
  AZURE_LOCATION \
  AZURE_RESOURCE_GROUP \
  AZURE_STORAGE_ACCOUNT_NAME \
  AZURE_BANK_1_CONTAINER_NAME \
  AZURE_BANK_2_CONTAINER_NAME \
  AZURE_BANK_3_CONTAINER_NAME \
  AZURE_OUTPUT_CONTAINER_NAME \
  CONTRACT_SERVICE_URL \
  CONTAINER_REGISTRY \
  TOOLS_HOME
do
  if [[ -z "${!required_var:-}" ]]; then
    echo "Environment variable $required_var not defined"
    exit 1
  fi
done

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

echo Obtaining contract service parameters...

validate_contract_service_url "$CONTRACT_SERVICE_URL"
export CONTRACT_SERVICE_PARAMETERS
CONTRACT_SERVICE_PARAMETERS=$(curl --fail --silent --show-error "${CONTRACT_SERVICE_URL%/}/parameters" | base64 --wrap=0)

echo Computing CCE policy...
envsubst < ../../policy/policy-in-template.json > /tmp/policy-in.json
export CCE_POLICY=$(az confcom acipolicygen -i /tmp/policy-in.json --debug-mode)
export CCE_POLICY_HASH=$(go run $TOOLS_HOME/securitypolicydigest/main.go -p $CCE_POLICY)
echo "Training container policy hash $CCE_POLICY_HASH"

export CONTRACTS=$contract
export PIPELINE_CONFIGURATION=`cat $pipelineConfiguration | base64 --wrap=0`

echo Generating encrypted file system information...
end=`date -u -d "60 minutes" '+%Y-%m-%dT%H:%MZ'`

BANK_1_SAS_TOKEN=$(az storage blob generate-sas --account-name $AZURE_STORAGE_ACCOUNT_NAME --container-name $AZURE_BANK_1_CONTAINER_NAME --permissions r --name data.img --expiry $end --only-show-errors)
export BANK_1_SAS_TOKEN=$(echo $BANK_1_SAS_TOKEN | tr -d \")
export BANK_1_SAS_TOKEN="?$BANK_1_SAS_TOKEN"

BANK_2_SAS_TOKEN=$(az storage blob generate-sas --account-name $AZURE_STORAGE_ACCOUNT_NAME --container-name $AZURE_BANK_2_CONTAINER_NAME --permissions r --name data.img --expiry $end --only-show-errors)
export BANK_2_SAS_TOKEN=$(echo $BANK_2_SAS_TOKEN | tr -d \")
export BANK_2_SAS_TOKEN="?$BANK_2_SAS_TOKEN"

BANK_3_SAS_TOKEN=$(az storage blob generate-sas --account-name $AZURE_STORAGE_ACCOUNT_NAME --container-name $AZURE_BANK_3_CONTAINER_NAME --permissions r --name data.img --expiry $end --only-show-errors)
export BANK_3_SAS_TOKEN=$(echo $BANK_3_SAS_TOKEN | tr -d \")
export BANK_3_SAS_TOKEN="?$BANK_3_SAS_TOKEN"

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
else
    echo "AZURE_KEYVAULT_ENDPOINT must end with .vault.azure.net or .managedhsm.azure.net"
    exit 1
fi

TMP=$(jq . encrypted-filesystem-config-template.json)
TMP=`echo $TMP | \
  jq '.azure_filesystems[0].azure_url = "https://" + env.AZURE_STORAGE_ACCOUNT_NAME + ".blob.core.windows.net/" + env.AZURE_BANK_1_CONTAINER_NAME + "/data.img" + env.BANK_1_SAS_TOKEN' | \
  jq '.azure_filesystems[0].mount_point = "/mnt/remote/bank_1"' | \
  jq '.azure_filesystems[0].key.kid = "Bank_1FilesystemEncryptionKey"' | \
  jq '.azure_filesystems[0].key.kty = env.AZURE_AKV_KEY_TYPE' | \
  jq '.azure_filesystems[0].key.akv.endpoint = env.AZURE_KEYVAULT_ENDPOINT' | \
  jq '.azure_filesystems[0].key.akv.bearer_token = env.BEARER_TOKEN' | \
  jq '.azure_filesystems[0].key_derivation.label = "Bank_1FilesystemEncryptionKey"' | \
  jq '.azure_filesystems[0].key_derivation.salt = "9b53cddbe5b78a0b912a8f05f341bcd4dd839ea85d26a08efaef13e696d999f4"'`

TMP=`echo $TMP | \
  jq '.azure_filesystems[1].azure_url = "https://" + env.AZURE_STORAGE_ACCOUNT_NAME + ".blob.core.windows.net/" + env.AZURE_BANK_2_CONTAINER_NAME + "/data.img" + env.BANK_2_SAS_TOKEN' | \
  jq '.azure_filesystems[1].mount_point = "/mnt/remote/bank_2"' | \
  jq '.azure_filesystems[1].key.kid = "Bank_2FilesystemEncryptionKey"' | \
  jq '.azure_filesystems[1].key.kty = env.AZURE_AKV_KEY_TYPE' | \
  jq '.azure_filesystems[1].key.akv.endpoint = env.AZURE_KEYVAULT_ENDPOINT' | \
  jq '.azure_filesystems[1].key.akv.bearer_token = env.BEARER_TOKEN' | \
  jq '.azure_filesystems[1].key_derivation.label = "Bank_2FilesystemEncryptionKey"' | \
  jq '.azure_filesystems[1].key_derivation.salt = "9b53cddbe5b78a0b912a8f05f341bcd4dd839ea85d26a08efaef13e696d999f4"'`

TMP=`echo $TMP | \
  jq '.azure_filesystems[2].azure_url = "https://" + env.AZURE_STORAGE_ACCOUNT_NAME + ".blob.core.windows.net/" + env.AZURE_BANK_3_CONTAINER_NAME + "/data.img" + env.BANK_3_SAS_TOKEN' | \
  jq '.azure_filesystems[2].mount_point = "/mnt/remote/bank_3"' | \
  jq '.azure_filesystems[2].key.kid = "Bank_3FilesystemEncryptionKey"' | \
  jq '.azure_filesystems[2].key.kty = env.AZURE_AKV_KEY_TYPE' | \
  jq '.azure_filesystems[2].key.akv.endpoint = env.AZURE_KEYVAULT_ENDPOINT' | \
  jq '.azure_filesystems[2].key.akv.bearer_token = env.BEARER_TOKEN' | \
  jq '.azure_filesystems[2].key_derivation.label = "Bank_3FilesystemEncryptionKey"' | \
  jq '.azure_filesystems[2].key_derivation.salt = "9b53cddbe5b78a0b912a8f05f341bcd4dd839ea85d26a08efaef13e696d999f4"'`

TMP=`echo $TMP | \
  jq '.azure_filesystems[3].azure_url = "https://" + env.AZURE_STORAGE_ACCOUNT_NAME + ".blob.core.windows.net/" + env.AZURE_OUTPUT_CONTAINER_NAME + "/data.img" + env.OUTPUT_SAS_TOKEN' | \
  jq '.azure_filesystems[3].mount_point = "/mnt/remote/output"' | \
  jq '.azure_filesystems[3].key.kid = "OutputFilesystemEncryptionKey"' | \
  jq '.azure_filesystems[3].key.kty = env.AZURE_AKV_KEY_TYPE' | \
  jq '.azure_filesystems[3].key.akv.endpoint = env.AZURE_KEYVAULT_ENDPOINT' | \
  jq '.azure_filesystems[3].key.akv.bearer_token = env.BEARER_TOKEN' | \
  jq '.azure_filesystems[3].key_derivation.label = "OutputFilesystemEncryptionKey"' | \
  jq '.azure_filesystems[3].key_derivation.salt = "9b53cddbe5b78a0b912a8f05f341bcd4dd839ea85d26a08efaef13e696d999f4"'`

ENCRYPTED_FILESYSTEM_INFORMATION=`echo $TMP | base64 --wrap=0`
echo $ENCRYPTED_FILESYSTEM_INFORMATION > /tmp/encrypted-filesystem-config.json
export ENCRYPTED_FILESYSTEM_INFORMATION

echo Generating parameters for ACI deployment...
TMP=$(jq '.containerRegistry.value = env.CONTAINER_REGISTRY' aci-parameters-template.json)
TMP=`echo $TMP | jq '.location.value = env.AZURE_LOCATION'`
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
