#!/bin/sh

set -e 

if [[ -z "${Contracts}" ]]; then 
  echo "Contract not specified"
  exit 1
fi

if [[ -z "${ContractServiceParameters}" ]]; then 
  echo "Contract service parameters not specified"
  exit 1
fi

if [[ -z "${ContractService}" ]]; then 
  echo "Contract service not specified"
  exit 1
fi

if [[ -z "${EncfsSideCarArgs}" ]]; then
  EncfsSideCarArgs=$1
fi

if [[ -z "${PipelineConfiguration}" ]]; then
  echo "Pipeline configuration not specified"
  exit 1
fi

echo EncfsSideCarArgs = $EncfsSideCarArgs

echo "Saving contract service parameters"
TRUST_STORE=/tmp/trust_store
mkdir -p $TRUST_STORE
echo $ContractServiceParameters | base64 -d > $TRUST_STORE/scitt.json

echo "Retrieving contract..."
scitt retrieve-contracts /tmp/contracts \
    --url ${ContractService} \
    --service-trust-store $TRUST_STORE \
    --from $Contracts \
    --development

ContractPayload="/tmp/contracts/2.$Contracts.json"
if [ ! -f $ContractPayload ]; then 
  echo "Contract does not exist" 
  exit 1
fi 

# Construct input by combining encrypted file system parameters and model configuration
echo $EncfsSideCarArgs | base64 -d > ./encfs.json
echo $PipelineConfiguration | base64 -d > ./pipeline_config.json
jq -s '.[0] * .[1]' ./encfs.json ./pipeline_config.json > config.json

echo "Checking contract..."
cat $ContractPayload

echo "Configuration..."
cat ./config.json

echo "Policy..."
cat ./policy.rego

# Check configuration against contract
 ./opa eval --fail -i ./config.json -d $ContractPayload -d ./policy.rego 'data.policy.allowed'

echo "Policy checked, mounting encrypted storage..."

if [[ -z "${EncfsSideCarArgs}" ]]; then
  if /bin/remotefs -logfile /log.txt -loglevel trace; then
    echo "1" > result
  else
    echo "0" > result
  fi
else
  if /bin/remotefs -logfile /log.txt -loglevel trace -base64 $EncfsSideCarArgs; then
    echo "1" > result
  else
    echo "0" > result
  fi
fi

echo "Writing training pipeline configuration..."

mkdir /mnt/remote/config
echo $PipelineConfiguration | base64 -d > /mnt/remote/config/pipeline_config.json

# Wait forever
while true; do sleep 1; done
