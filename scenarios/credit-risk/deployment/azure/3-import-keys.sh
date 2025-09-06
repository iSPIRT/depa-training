#!/bin/bash

# Function to import a key with a given key ID and key material into AKV
# The key is bound to a key release policy with host data defined in the environment variable CCE_POLICY_HASH
function import_key() {
  export KEYID=$1
  export KEYFILE=$2

  # For RSA-HSM keys, we need to set a salt and label which will be used in the symmetric key derivation
  if [ "$AZURE_AKV_KEY_TYPE" = "RSA-HSM" ]; then    
    export AZURE_AKV_KEY_DERIVATION_LABEL=$KEYID
  fi

  CONFIG=$(jq '.claims[0][0].equals = env.CCE_POLICY_HASH' importkey-config-template.json)
  CONFIG=$(echo $CONFIG | jq '.key.kid = env.KEYID')
  CONFIG=$(echo $CONFIG | jq '.key.kty = env.AZURE_AKV_KEY_TYPE')
  CONFIG=$(echo $CONFIG | jq '.key_derivation.salt = "9b53cddbe5b78a0b912a8f05f341bcd4dd839ea85d26a08efaef13e696d999f4"')
  CONFIG=$(echo $CONFIG | jq '.key_derivation.label = env.AZURE_AKV_KEY_DERIVATION_LABEL')
  CONFIG=$(echo $CONFIG | jq '.key.akv.endpoint = env.AZURE_KEYVAULT_ENDPOINT')
  CONFIG=$(echo $CONFIG | jq '.key.akv.bearer_token = env.BEARER_TOKEN')
  echo $CONFIG > /tmp/importkey-config.json
  echo "Importing $KEYID key with key release policy"
  jq '.key.akv.bearer_token = "REDACTED"' /tmp/importkey-config.json
  pushd . && cd $TOOLS_HOME/importkey && go run main.go -c /tmp/importkey-config.json -out && popd
  mv $TOOLS_HOME/importkey/keyfile.bin $KEYFILE
}

echo Obtaining contract service parameters...
CONTRACT_SERVICE_URL=${CONTRACT_SERVICE_URL:-"http://localhost:8000"}
export CONTRACT_SERVICE_PARAMETERS=$(curl -k -f $CONTRACT_SERVICE_URL/parameters | base64 --wrap=0)

envsubst < ../../policy/policy-in-template.json > /tmp/policy-in.json
export CCE_POLICY=$(az confcom acipolicygen -i /tmp/policy-in.json --debug-mode)
export CCE_POLICY_HASH=$(go run $TOOLS_HOME/securitypolicydigest/main.go -p $CCE_POLICY)
echo "Training container policy hash $CCE_POLICY_HASH"

# Obtain the token based on the AKV resource endpoint subdomain
if [[ "$AZURE_KEYVAULT_ENDPOINT" == *".vault.azure.net" ]]; then
    export BEARER_TOKEN=$(az account get-access-token --resource https://vault.azure.net | jq -r .accessToken)
    echo "Importing keys to AKV key vaults can be only of type RSA-HSM"
    export AZURE_AKV_KEY_TYPE="RSA-HSM"
elif [[ "$AZURE_KEYVAULT_ENDPOINT" == *".managedhsm.azure.net" ]]; then
    export BEARER_TOKEN=$(az account get-access-token --resource https://managedhsm.azure.net | jq -r .accessToken)    
    export AZURE_AKV_KEY_TYPE="oct-HSM"
fi

DATADIR=$REPO_ROOT/scenarios/$SCENARIO/data
MODELDIR=$REPO_ROOT/scenarios/$SCENARIO/modeller

import_key "BankAFilesystemEncryptionKey" $DATADIR/bank_a_key.bin
import_key "BankBFilesystemEncryptionKey" $DATADIR/bank_b_key.bin
import_key "BureauFilesystemEncryptionKey" $DATADIR/bureau_key.bin
import_key "FintechFilesystemEncryptionKey" $DATADIR/fintech_key.bin
import_key "OutputFilesystemEncryptionKey" $MODELDIR/output_key.bin

## Cleanup
rm /tmp/importkey-config.json
rm /tmp/policy-in.json
