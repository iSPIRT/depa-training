#!/bin/bash

# Usage:
#   ./create-scenario.sh <path-to-scenario.json> [--force]

if ! command -v jq >/dev/null 2>&1; then
  echo "Error: jq is required. Install with: sudo apt-get install -y jq" >&2
  exit 1
fi

if [ $# -lt 1 ]; then
  echo "Usage: $0 <path-to-scenario.json> [--force]" >&2
  exit 1
fi

INPUT_JSON="$1"
FORCE="false"
if [ "${2-}" = "--force" ]; then
  FORCE="true"
fi

if [ ! -f "$INPUT_JSON" ]; then
  echo "Error: JSON file not found: $INPUT_JSON" >&2
  exit 1
fi

REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"

# Extract scenario name
SCENARIO_NAME="$(jq -r '.scenario_name' "$INPUT_JSON")"
if [ -z "$SCENARIO_NAME" ] || [ "$SCENARIO_NAME" = "null" ]; then
  echo "Error: scenario_name missing in $INPUT_JSON" >&2
  exit 1
fi

SCENARIO_DIR="$REPO_ROOT/scenarios/$SCENARIO_NAME"
if [ -d "$SCENARIO_DIR" ] && [ "$FORCE" != "true" ]; then
  echo "Error: Scenario directory already exists: $SCENARIO_DIR (use --force to overwrite)" >&2
  exit 1
fi

# Extract all TDP names
TDP_NAMES=($(jq -r '.tdps[] | .name' "$INPUT_JSON"))
if [ -z "$TDP_NAMES" ] || [ "$TDP_NAMES" = "null" ]; then
  echo "Error: tdp_names missing in $INPUT_JSON" >&2
  exit 1
fi

echo "Creating scenario: $SCENARIO_NAME"
mkdir -p "$SCENARIO_DIR"

# Basic layout
mkdir -p "$SCENARIO_DIR/ci"
mkdir -p "$SCENARIO_DIR/src"
mkdir -p "$SCENARIO_DIR/policy"
mkdir -p "$SCENARIO_DIR/contract"
mkdir -p "$SCENARIO_DIR/config/templates"
mkdir -p "$SCENARIO_DIR/deployment/azure"
mkdir -p "$SCENARIO_DIR/deployment/local"


######## Create ci and src ########

MODEL_FORMAT="$(jq -r '.model_format' "$INPUT_JSON")"
TRAIN_METHOD="$(jq -r '.training_framework' "$INPUT_JSON")"
JOIN_TYPE="$(jq -r '.join_type' "$INPUT_JSON")"

# Generate Dockerfiles and preprocess scripts for each TDP
for TDP in "${TDP_NAMES[@]}"; do
  CLEAN_NAME=$(echo "$TDP" | tr -d '_' | tr -d '-')  # remove _ and -
  DOCKERFILE="$SCENARIO_DIR/ci/Dockerfile.${CLEAN_NAME}"
  PREPROCESS="$SCENARIO_DIR/src/preprocess_${TDP}.py"

  # Generate Dockerfile
  cat > "$DOCKERFILE" <<EOF
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND="noninteractive"

RUN apt-get update && apt-get -y upgrade \\
    && apt-get install -y curl \\
    && apt-get install -y python3 python3-dev python3-distutils 

## Install pip
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
RUN python3 get-pip.py

## Install dependencies
# RUN pip3 install

COPY preprocess_${TDP}.py preprocess_${TDP}.py
EOF

  # Generate preprocess script
  cat > "$PREPROCESS" <<EOF
# TODO: Add your library imports here
import os

input_dir = "/mnt/input/data"
output_dir = "/mnt/output/preprocessed"
os.makedirs(output_dir, exist_ok=True)

# TODO: Implement your data processing code here

EOF
done

if [ "$MODEL_FORMAT" != "ccr_instantiate" ]; then
    # Generate Dockerfile.modelsave
    cat > "$SCENARIO_DIR/ci/Dockerfile.modelsave" <<EOF
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND="noninteractive"

RUN apt-get update && apt-get -y upgrade \\
    && apt-get install -y curl \\
    && apt-get install -y python3 python3-dev python3-distutils 

## Install pip
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
RUN python3 get-pip.py

## Install dependencies
# RUN pip3 install

COPY save_base_model.py save_base_model.py
EOF

    # Generate save_base_model.py
    cat > "$SCENARIO_DIR/src/save_base_model.py" <<EOF
# TODO: Add your library imports here
import os

save_dir = "/mnt/model"
os.makedirs(save_dir, exist_ok=True)

# TODO: Implement model instantiation and saving logic here

EOF
fi

# Generate build.sh
BUILD_SCRIPT="$SCENARIO_DIR/ci/build.sh"
echo '#!/bin/bash' > "$BUILD_SCRIPT"
for TDP in "${TDP_NAMES[@]}"; do
  # remove _ and -
  CLEAN_NAME=$(echo "$TDP" | tr -d '_' | tr -d '-')
  # replace _ with -
  TDP_SUFFIX=$(echo "$TDP" | tr '_' '-' | tr '[:upper:]' '[:lower:]')
  SCENARIO_PREFIX=$(echo "$SCENARIO_NAME" | tr '_' '-')
  echo "docker build -f ci/Dockerfile.${CLEAN_NAME} src -t preprocess-${TDP_SUFFIX}:latest" >> "$BUILD_SCRIPT"
done
if [ "$MODEL_FORMAT" != "ccr_instantiate" ]; then
    echo "docker build -f ci/Dockerfile.modelsave src -t ${SCENARIO_PREFIX}-model-save:latest" >> "$BUILD_SCRIPT"
fi
chmod +x "$BUILD_SCRIPT"

# Generate pull-containers.sh
PULL_SCRIPT="$SCENARIO_DIR/ci/pull-containers.sh"
echo '#!/bin/bash' > "$PULL_SCRIPT"
for TDP in "${TDP_NAMES[@]}"; do
  CLEAN_NAME=$(echo "$TDP" | tr -d '_' | tr -d '-')
  TDP_SUFFIX=$(echo "$CLEAN_NAME" | tr '_' '-' | tr '[:upper:]' '[:lower:]')
  echo "docker pull \$CONTAINER_REGISTRY/preprocess-${TDP_SUFFIX}:latest" >> "$PULL_SCRIPT"
done
if [ "$MODEL_FORMAT" != "ccr_instantiate" ]; then
    echo "docker pull \$CONTAINER_REGISTRY/${SCENARIO_PREFIX}-model-save:latest" >> "$PULL_SCRIPT"
fi
chmod +x "$PULL_SCRIPT"

# Generate push-containers.sh
PUSH_SCRIPT="$SCENARIO_DIR/ci/push-containers.sh"
echo '#!/bin/bash' > "$PUSH_SCRIPT"
for TDP in "${TDP_NAMES[@]}"; do
  CLEAN_NAME=$(echo "$TDP" | tr -d '_' | tr -d '-')
  TDP_SUFFIX=$(echo "$CLEAN_NAME" | tr '_' '-' | tr '[:upper:]' '[:lower:]')
  echo "docker tag preprocess-${TDP_SUFFIX}:latest \$CONTAINER_REGISTRY/preprocess-${TDP_SUFFIX}:latest" >> "$PUSH_SCRIPT"
  echo "docker push \$CONTAINER_REGISTRY/preprocess-${TDP_SUFFIX}:latest" >> "$PUSH_SCRIPT"
done
if [ "$MODEL_FORMAT" != "ccr_instantiate" ]; then
    echo "docker tag ${SCENARIO_PREFIX}-model-save:latest \$CONTAINER_REGISTRY/${SCENARIO_PREFIX}-model-save:latest" >> "$PUSH_SCRIPT"
    echo "docker push \$CONTAINER_REGISTRY/${SCENARIO_PREFIX}-model-save:latest" >> "$PUSH_SCRIPT"
fi

chmod +x "$PUSH_SCRIPT"


######## Create contract ########

OUTPUT_FILE="$SCENARIO_DIR/contract/contract.json"

# Random contract UUID
CONTRACT_ID=$(uuidgen)

# Begin contract object
jq -n \
  --arg id "$CONTRACT_ID" \
  --arg schemaVersion "0.1" \
  --arg startTime "2025-09-14T00:00:00.000Z" \
  --arg expiryTime "2025-10-14T00:00:00.000Z" \
  --arg purpose "TRAINING" \
  '
  {
    id: $id,
    schemaVersion: $schemaVersion,
    startTime: $startTime,
    expiryTime: $expiryTime,
    tdc: "",
    tdps: [],
    ccrp: "did:web:$CCRP_USERNAME.github.io",
    datasets: [],
    purpose: $purpose,
    constraints: [ { privacy: [] } ],
    terms: { payment: {}, revocation: {} }
  }
  ' > "$OUTPUT_FILE"

# Iterate datasets and append
jq -c '.tdps[] | {tdp: .name, datasets: .datasets}' "$INPUT_JSON" | while read -r entry; do
  TDP_NAME=$(echo "$entry" | jq -r '.tdp')
  DATASETS=$(echo "$entry" | jq -c '.datasets[]')
  
  for ds in $DATASETS; do
    DS_ID=$(echo "$ds" | jq -r '.id')
    DS_NAME=$(echo "$ds" | jq -r '.name')

    # Epsilon and delta are optional
    EPS=$([[ "$ds" =~ .*epsilon.* ]] && echo "$ds" | jq -r '.epsilon' || echo "null") # if epsilon is present, use it, otherwise set to null
    DELTA=$([[ "$ds" =~ .*delta.* ]] && echo "$ds" | jq -r '.delta' || echo "null") # if delta is present, use it, otherwise set to null

    # Uppercase container name
    UPPER_NAME=$(echo "$TDP_NAME" | tr '[:lower:]' '[:upper:]')

    # Key kid (BankA → BankAFilesystemEncryptionKey)
    KEY_KID="$(echo "$TDP_NAME" | sed -E 's/(^|_)([a-z])/\U\2/g')FilesystemEncryptionKey"

    # Dataset JSON
    DATASET_OBJ=$(jq -n \
      --arg id "$DS_ID" \
      --arg name "$DS_NAME" \
      --arg owner "$TDP_NAME" \
      --arg url "https://\$AZURE_STORAGE_ACCOUNT_NAME.blob.core.windows.net/\$AZURE_${UPPER_NAME}_CONTAINER_NAME/data.img" \
      --arg provider "" \
      --arg kid "$KEY_KID" \
      '
      {
        id: $id,
        name: $name,
        owner: $owner,
        url: $url,
        provider: $provider,
        key: {
          type: "azure",
          properties: {
            kid: $kid,
            authority: { endpoint: "sharedneu.neu.attest.azure.net" },
            endpoint: ""
          }
        }
      }
      ')

    # Privacy JSON
    PRIVACY_OBJ=$(jq -n \
      --arg dataset "$DS_ID" \
      --arg eps "$EPS" \
      --arg delta "$DELTA" \
      '{ dataset: $dataset, epsilon_threshold: $eps, delta: $delta }')

    # Append into contract.json
    tmp=$(mktemp)
    jq \
      --argjson dataset "$DATASET_OBJ" \
      --argjson privacy "$PRIVACY_OBJ" \
      '.datasets += [$dataset] | .constraints[0].privacy += [$privacy]' \
      "$OUTPUT_FILE" > "$tmp" && mv "$tmp" "$OUTPUT_FILE"
  done
done



######## Create policy ########

POLICY_FILE="$SCENARIO_DIR/policy/policy-in-template.json"

jq -n \
  '
  {
    version: "1.0",
    containers: [
      {
        containerImage: "$CONTAINER_REGISTRY/depa-training:latest",
        command: [
          "/bin/bash",
          "run.sh"
        ],
        environmentVariables: [],
        mounts: [
          {
            mountType: "emptyDir",
            mountPath: "/mnt/remote",
            readonly: false
          }
        ]
      },
      {
        containerImage: "$CONTAINER_REGISTRY/depa-training-encfs:latest",
        environmentVariables: [
          {
            name: "EncfsSideCarArgs",
            value: ".+",
            strategy: "re2"
          },
          {
            name: "ContractService",
            value: ".+",
            strategy: "re2"
          },
          {
            name: "ContractServiceParameters",
            value: "$CONTRACT_SERVICE_PARAMETERS",
            strategy: "string"
          },
          {
            name: "Contracts",
            value: ".+",
            strategy: "re2"
          },
          {
            name: "PipelineConfiguration",
            value: ".+",
            strategy: "re2"
          }
        ],
        command: [
          "/encfs.sh"
        ],
        securityContext: {
          privileged: "true"
        },
        mounts: [
          {
            mountType: "emptyDir",
            mountPath: "/mnt/remote",
            readonly: false
          }
        ]
      }
    ]
  }
  ' > "$POLICY_FILE"


######## Create deployment/azure ########

# 1-create-storage-containers.sh

AZ_SCRIPT_1="$SCENARIO_DIR/deployment/azure/1-create-storage-containers.sh"

echo '#!/bin/bash

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
  echo "Storage account $AZURE_STORAGE_ACCOUNT_NAME already exists. Skipping creation."
fi

# Get the storage account key
ACCOUNT_KEY=$(az storage account keys list --resource-group $AZURE_RESOURCE_GROUP --account-name $AZURE_STORAGE_ACCOUNT_NAME --query "[0].value" --output tsv)

' > "$AZ_SCRIPT_1"

TDP_NAMES=($(jq -r '.tdps[] | .name' "$INPUT_JSON"))
for TDP in "${TDP_NAMES[@]}"; do
  UPPER_NAME=$(echo "$TDP" | tr '[:lower:]' '[:upper:]')
  cat >> "$AZ_SCRIPT_1" <<EOF
# Check if the ${UPPER_NAME} container exists
CONTAINER_EXISTS=\$(az storage container exists --name \$AZURE_${UPPER_NAME}_CONTAINER_NAME --account-name \$AZURE_STORAGE_ACCOUNT_NAME --account-key \$ACCOUNT_KEY --query "exists" --output tsv)

if [ "\$CONTAINER_EXISTS" == "false" ]; then
    echo "Container \$AZURE_${UPPER_NAME}_CONTAINER_NAME does not exist. Creating it now..."
    az storage container create --resource-group \$AZURE_RESOURCE_GROUP --account-name \$AZURE_STORAGE_ACCOUNT_NAME --name \$AZURE_${UPPER_NAME}_CONTAINER_NAME --account-key \$ACCOUNT_KEY
fi

EOF
done


if [ "$MODEL_FORMAT" != "ccr_instantiate" ]; then
    cat >> "$AZ_SCRIPT_1" <<'EOF'
# Check if the MODEL container exists
CONTAINER_EXISTS=$(az storage container exists --name $AZURE_MODEL_CONTAINER_NAME --account-name $AZURE_STORAGE_ACCOUNT_NAME --account-key $ACCOUNT_KEY --query "exists" --output tsv)

if [ "$CONTAINER_EXISTS" == "false" ]; then
echo "Container $AZURE_MODEL_CONTAINER_NAME does not exist. Creating it now..."
az storage container create --resource-group $AZURE_RESOURCE_GROUP --account-name $AZURE_STORAGE_ACCOUNT_NAME --name $AZURE_MODEL_CONTAINER_NAME --account-key $ACCOUNT_KEY
fi

EOF
fi

cat >> "$AZ_SCRIPT_1" <<'EOF'
# Check if the OUTPUT container exists
CONTAINER_EXISTS=$(az storage container exists --name $AZURE_OUTPUT_CONTAINER_NAME --account-name $AZURE_STORAGE_ACCOUNT_NAME --account-key $ACCOUNT_KEY --query "exists" --output tsv)

if [ "$CONTAINER_EXISTS" == "false" ]; then
  echo "Container $AZURE_OUTPUT_CONTAINER_NAME does not exist. Creating it now..."
  az storage container create --resource-group $AZURE_RESOURCE_GROUP --account-name $AZURE_STORAGE_ACCOUNT_NAME --name $AZURE_OUTPUT_CONTAINER_NAME --account-key $ACCOUNT_KEY
fi
EOF

chmod +x "$AZ_SCRIPT_1"


# 2-create-akv.sh

AZ_SCRIPT_2="$SCENARIO_DIR/deployment/azure/2-create-akv.sh"

cat > "$AZ_SCRIPT_2" <<EOF
#!/bin/bash

set -e 

if [[ "\$AZURE_KEYVAULT_ENDPOINT" == *".vault.azure.net" ]]; then 
    AZURE_AKV_RESOURCE_NAME=\`echo \$AZURE_KEYVAULT_ENDPOINT | awk '{split(\$0,a,"."); print a[1]}'\`
    # Check if the Key Vault already exists
    echo "Checking if Key Vault \$AZURE_AKV_RESOURCE_NAME exists..."
    NAME_AVAILABLE=\$(az rest --method post \
        --uri "https://management.azure.com/subscriptions/\$AZURE_SUBSCRIPTION_ID/providers/Microsoft.KeyVault/checkNameAvailability?api-version=2019-09-01" \
        --headers "Content-Type=application/json" \
        --body "{\"name\": \"\$AZURE_AKV_RESOURCE_NAME\", \"type\": \"Microsoft.KeyVault/vaults\"}" | jq -r '.nameAvailable')
    if [ "\$NAME_AVAILABLE" == true ]; then
        echo "Key Vault \$AZURE_AKV_RESOURCE_NAME does not exist. Creating it now..."
        echo CREATING \$AZURE_KEYVAULT_ENDPOINT in resource group \$AZURE_RESOURCE_GROUP
        # Create Azure key vault with RBAC authorization
        az keyvault create --name \$AZURE_AKV_RESOURCE_NAME --resource-group \$AZURE_RESOURCE_GROUP --sku "Premium" --enable-rbac-authorization
        # Assign RBAC roles to the resource owner so they can import keys
        AKV_SCOPE=\`az keyvault show --name \$AZURE_AKV_RESOURCE_NAME --query id --output tsv\`    
        az role assignment create --role "Key Vault Crypto Officer" --assignee \`az account show --query user.name --output tsv\` --scope \$AKV_SCOPE
        az role assignment create --role "Key Vault Crypto User" --assignee \`az account show --query user.name --output tsv\` --scope \$AKV_SCOPE
    else
        # Name is not available — check if the vault exists in *this* subscription
        if az keyvault show --name \$AZURE_AKV_RESOURCE_NAME --resource-group \$AZURE_RESOURCE_GROUP >/dev/null 2>&1; then
            echo "Key Vault \$AZURE_AKV_RESOURCE_NAME already exists in your subscription. Skipping creation."
        else
            echo "Key Vault name '\$AZURE_AKV_RESOURCE_NAME' is already reserved."
            echo "If you previously owned this Key Vault, it may be soft-deleted."
            echo "Please purge the Key Vault before retrying."
            exit 1
        fi
    fi
else
    echo "Automated creation of key vaults is supported only for vaults"
fi

EOF

chmod +x "$AZ_SCRIPT_2"


# 3-import-keys.sh

AZ_SCRIPT_3="$SCENARIO_DIR/deployment/azure/3-import-keys.sh"

cat > "$AZ_SCRIPT_3" <<EOF
#!/bin/bash

# Function to import a key with a given key ID and key material into AKV
# The key is bound to a key release policy with host data defined in the environment variable CCE_POLICY_HASH
function import_key() {
  export KEYID=\$1
  export KEYFILE=\$2

  # For RSA-HSM keys, we need to set a salt and label which will be used in the symmetric key derivation
  if [ "\$AZURE_AKV_KEY_TYPE" = "RSA-HSM" ]; then    
    export AZURE_AKV_KEY_DERIVATION_LABEL=\$KEYID
  fi

  CONFIG=\$(jq '.claims[0][0].equals = env.CCE_POLICY_HASH' importkey-config-template.json)
  CONFIG=\$(echo \$CONFIG | jq '.key.kid = env.KEYID')
  CONFIG=\$(echo \$CONFIG | jq '.key.kty = env.AZURE_AKV_KEY_TYPE')
  CONFIG=\$(echo \$CONFIG | jq '.key_derivation.salt = "9b53cddbe5b78a0b912a8f05f341bcd4dd839ea85d26a08efaef13e696d999f4"')
  CONFIG=\$(echo \$CONFIG | jq '.key_derivation.label = env.AZURE_AKV_KEY_DERIVATION_LABEL')
  CONFIG=\$(echo \$CONFIG | jq '.key.akv.endpoint = env.AZURE_KEYVAULT_ENDPOINT')
  CONFIG=\$(echo \$CONFIG | jq '.key.akv.bearer_token = env.BEARER_TOKEN')
  echo \$CONFIG > /tmp/importkey-config.json
  echo "Importing \$KEYID key with key release policy"
  jq '.key.akv.bearer_token = "REDACTED"' /tmp/importkey-config.json
  pushd . && cd \$TOOLS_HOME/importkey && go run main.go -c /tmp/importkey-config.json -out && popd
  mv \$TOOLS_HOME/importkey/keyfile.bin \$KEYFILE
}

echo Obtaining contract service parameters...
CONTRACT_SERVICE_URL=\${CONTRACT_SERVICE_URL:-"http://localhost:8000"}
export CONTRACT_SERVICE_PARAMETERS=\$(curl -k -f \$CONTRACT_SERVICE_URL/parameters | base64 --wrap=0)

envsubst < ../../policy/policy-in-template.json > /tmp/policy-in.json
export CCE_POLICY=\$(az confcom acipolicygen -i /tmp/policy-in.json --debug-mode)
export CCE_POLICY_HASH=\$(go run \$TOOLS_HOME/securitypolicydigest/main.go -p \$CCE_POLICY)
echo "Training container policy hash \$CCE_POLICY_HASH"

# Obtain the token based on the AKV resource endpoint subdomain
if [[ "\$AZURE_KEYVAULT_ENDPOINT" == *".vault.azure.net" ]]; then
    export BEARER_TOKEN=\$(az account get-access-token --resource https://vault.azure.net | jq -r .accessToken)
    echo "Importing keys to AKV key vaults can be only of type RSA-HSM"
    export AZURE_AKV_KEY_TYPE="RSA-HSM"
elif [[ "\$AZURE_KEYVAULT_ENDPOINT" == *".managedhsm.azure.net" ]]; then
    export BEARER_TOKEN=\$(az account get-access-token --resource https://managedhsm.azure.net | jq -r .accessToken)    
    export AZURE_AKV_KEY_TYPE="oct-HSM"
fi

DATADIR=\$REPO_ROOT/scenarios/\$SCENARIO/data
MODELDIR=\$REPO_ROOT/scenarios/\$SCENARIO/modeller

EOF

for TDP in "${TDP_NAMES[@]}"; do
  KEY_KID="$(echo "$TDP" | sed -E 's/(^|_)([a-z])/\U\2/g')FilesystemEncryptionKey"
  cat >> "$AZ_SCRIPT_3" <<EOF
import_key "$KEY_KID" \$DATADIR/${TDP}_key.bin
EOF
done

if [ "$MODEL_FORMAT" != "ccr_instantiate" ]; then
    cat >> "$AZ_SCRIPT_3" <<EOF
import_key "ModelFilesystemEncryptionKey" \$MODELDIR/model_key.bin
EOF
fi

cat >> "$AZ_SCRIPT_3" <<EOF
import_key "OutputFilesystemEncryptionKey" \$MODELDIR/output_key.bin

## Cleanup
rm /tmp/importkey-config.json
rm /tmp/policy-in.json
EOF

chmod +x "$AZ_SCRIPT_3"


# 4-encrypt-data.sh

AZ_SCRIPT_4="$SCENARIO_DIR/deployment/azure/4-encrypt-data.sh"
cat > "$AZ_SCRIPT_4" <<EOF
#!/bin/bash

DATADIR=\$REPO_ROOT/scenarios/\$SCENARIO/data
MODELDIR=\$REPO_ROOT/scenarios/\$SCENARIO/modeller

EOF

for TDP in "${TDP_NAMES[@]}"; do
  cat >> "$AZ_SCRIPT_4" <<EOF
./generatefs.sh -d \$DATADIR/$TDP/preprocessed -k \$DATADIR/${TDP}_key.bin -i \$DATADIR/${TDP}.img
EOF
done

if [ "$MODEL_FORMAT" != "ccr_instantiate" ]; then
    cat >> "$AZ_SCRIPT_4" <<EOF

./generatefs.sh -d \$MODELDIR/models -k \$MODELDIR/model_key.bin -i \$MODELDIR/model.img
EOF
fi

cat >> "$AZ_SCRIPT_4" <<EOF

sudo rm -rf \$MODELDIR/output
mkdir -p \$MODELDIR/output
./generatefs.sh -d \$MODELDIR/output -k \$MODELDIR/output_key.bin -i \$MODELDIR/output.img
EOF

chmod +x "$AZ_SCRIPT_4"


# 5-upload-encrypted-data.sh

AZ_SCRIPT_5="$SCENARIO_DIR/deployment/azure/5-upload-encrypted-data.sh"
cat > "$AZ_SCRIPT_5" <<EOF
#!/bin/bash

DATADIR=\$REPO_ROOT/scenarios/\$SCENARIO/data
MODELDIR=\$REPO_ROOT/scenarios/\$SCENARIO/modeller

ACCOUNT_KEY=\$(az storage account keys list --account-name \$AZURE_STORAGE_ACCOUNT_NAME --only-show-errors | jq -r .[0].value)

EOF

for TDP in "${TDP_NAMES[@]}"; do
  UPPER_NAME=$(echo "$TDP" | tr '[:lower:]' '[:upper:]')
  cat >> "$AZ_SCRIPT_5" <<EOF
az storage blob upload \\
  --account-name \$AZURE_STORAGE_ACCOUNT_NAME \\
  --container \$AZURE_${UPPER_NAME}_CONTAINER_NAME \\
  --file \$DATADIR/${TDP}.img \\
  --name data.img \\
  --type page \\
  --overwrite \\
  --account-key \$ACCOUNT_KEY

EOF
done

if [ "$MODEL_FORMAT" != "ccr_instantiate" ]; then
    cat >> "$AZ_SCRIPT_5" <<EOF
az storage blob upload \\
  --account-name \$AZURE_STORAGE_ACCOUNT_NAME \\
  --container \$AZURE_MODEL_CONTAINER_NAME \\
  --file \$MODELDIR/model.img \\
  --name data.img \\
  --type page \\
  --overwrite \\
  --account-key \$ACCOUNT_KEY

EOF
fi

cat >> "$AZ_SCRIPT_5" <<EOF
az storage blob upload \\
  --account-name \$AZURE_STORAGE_ACCOUNT_NAME \\
  --container \$AZURE_OUTPUT_CONTAINER_NAME \\
  --file \$MODELDIR/output.img \\
  --name data.img \\
  --type page \\
  --overwrite \\
  --account-key \$ACCOUNT_KEY
EOF

chmod +x "$AZ_SCRIPT_5"


# 6-download-decrypt-model.sh

AZ_SCRIPT_6="$SCENARIO_DIR/deployment/azure/6-download-decrypt-model.sh"
cat > "$AZ_SCRIPT_6" <<EOF
#!/bin/bash

MODELDIR=\$REPO_ROOT/scenarios/\$SCENARIO/modeller

sudo rm -rf \$MODELDIR/output
mkdir -p \$MODELDIR/output

ACCOUNT_KEY=\$(az storage account keys list --account-name \$AZURE_STORAGE_ACCOUNT_NAME --only-show-errors | jq -r .[0].value)

az storage blob download \\
  --account-name \$AZURE_STORAGE_ACCOUNT_NAME \\
  --container \$AZURE_OUTPUT_CONTAINER_NAME \\
  --file \$MODELDIR/output.img \\
  --name data.img \\
  --account-key \$ACCOUNT_KEY

encryptedImage=\$MODELDIR/output.img
keyFilePath=\$MODELDIR/output_key.bin

echo Decrypting \$encryptedImage with key \$keyFilePath
deviceName=cryptdevice1
deviceNamePath="/dev/mapper/\$deviceName"

sudo cryptsetup luksOpen "\$encryptedImage" "\$deviceName" \\
    --key-file "\$keyFilePath" \\ 
    --integrity-no-journal --persistent

mountPoint=\`mktemp -d\`
sudo mount -t ext4 "\$deviceNamePath" "\$mountPoint" -o loop

cp -r \$mountPoint/* \$MODELDIR/output/

echo "[!] Closing device..."

sudo umount "\$mountPoint"
sleep 2
sudo cryptsetup luksClose "\$deviceName"
EOF

chmod +x "$AZ_SCRIPT_6"


# deploy.sh

AZ_SCRIPT_7="$SCENARIO_DIR/deployment/azure/deploy.sh"
cat > "$AZ_SCRIPT_7" <<EOF
#!/bin/bash

set -e 

while getopts ":c:p:" options; do
    case \$options in 
        c)contract=\$OPTARG;;
        p)pipelineConfiguration=\$OPTARG;;
    esac
done

if [[ -z "\${contract}" ]]; then
  echo "No contract specified"
  exit 1
fi

if [[ -z "\${pipelineConfiguration}" ]]; then
  echo "No pipeline configuration specified"
  exit 1
fi

if [[ -z "\${AZURE_KEYVAULT_ENDPOINT}" ]]; then
  echo "Environment variable AZURE_KEYVAULT_ENDPOINT not defined"
fi

echo Obtaining contract service parameters...

CONTRACT_SERVICE_URL=\${CONTRACT_SERVICE_URL:-"https://localhost:8000"}
export CONTRACT_SERVICE_PARAMETERS=\$(curl -k -f \$CONTRACT_SERVICE_URL/parameters | base64 --wrap=0)

echo Computing CCE policy...
envsubst < ../../policy/policy-in-template.json > /tmp/policy-in.json
export CCE_POLICY=\$(az confcom acipolicygen -i /tmp/policy-in.json --debug-mode)
export CCE_POLICY_HASH=\$(go run \$TOOLS_HOME/securitypolicydigest/main.go -p \$CCE_POLICY)
echo "Training container policy hash \$CCE_POLICY_HASH"

export CONTRACTS=\$contract
export PIPELINE_CONFIGURATION=\`cat \$pipelineConfiguration | base64 --wrap=0\`

echo Generating encrypted file system information...
end=\`date -u -d "60 minutes" '+%Y-%m-%dT%H:%MZ'\`

EOF
for TDP in "${TDP_NAMES[@]}"; do
  UPPER_NAME=$(echo "$TDP" | tr '[:lower:]' '[:upper:]')
  TOKEN_NAME="${UPPER_NAME}_SAS_TOKEN"
  cat >> "$AZ_SCRIPT_7" <<EOF
${TOKEN_NAME}=\$(az storage blob generate-sas --account-name \$AZURE_STORAGE_ACCOUNT_NAME --container-name \$AZURE_${UPPER_NAME}_CONTAINER_NAME --permissions r --name data.img --expiry \$end --only-show-errors) 
export ${TOKEN_NAME}=\$(echo \$${TOKEN_NAME} | tr -d \")
export ${TOKEN_NAME}="?\$${TOKEN_NAME}"

EOF
done

if [ "$MODEL_FORMAT" != "ccr_instantiate" ]; then
    cat >> "$AZ_SCRIPT_7" <<EOF
MODEL_SAS_TOKEN=\$(az storage blob generate-sas --account-name \$AZURE_STORAGE_ACCOUNT_NAME --container-name \$AZURE_MODEL_CONTAINER_NAME --permissions r --name data.img --expiry \$end --only-show-errors) 
export MODEL_SAS_TOKEN=\$(echo \$MODEL_SAS_TOKEN | tr -d \")
export MODEL_SAS_TOKEN="?\$MODEL_SAS_TOKEN"

EOF
fi

cat >> "$AZ_SCRIPT_7" <<EOF
OUTPUT_SAS_TOKEN=\$(az storage blob generate-sas --account-name \$AZURE_STORAGE_ACCOUNT_NAME --container-name \$AZURE_OUTPUT_CONTAINER_NAME --permissions rw --name data.img --expiry \$end --only-show-errors) 
export OUTPUT_SAS_TOKEN=\$(echo \$OUTPUT_SAS_TOKEN | tr -d \")
export OUTPUT_SAS_TOKEN="?\$OUTPUT_SAS_TOKEN"

# Obtain the token based on the AKV resource endpoint subdomain
if [[ "\$AZURE_KEYVAULT_ENDPOINT" == *".vault.azure.net" ]]; then
    export BEARER_TOKEN=\$(az account get-access-token --resource https://vault.azure.net | jq -r .accessToken)
    echo "Importing keys to AKV key vaults can be only of type RSA-HSM"
    export AZURE_AKV_KEY_TYPE="RSA-HSM"
elif [[ "\$AZURE_KEYVAULT_ENDPOINT" == *".managedhsm.azure.net" ]]; then
    export BEARER_TOKEN=\$(az account get-access-token --resource https://managedhsm.azure.net | jq -r .accessToken)
    export AZURE_AKV_KEY_TYPE="oct-HSM"
fi

TMP=\$(jq . encrypted-filesystem-config-template.json)
EOF

idx=0
for TDP in "${TDP_NAMES[@]}"; do
  UPPER_NAME=$(echo "$TDP" | tr '[:lower:]' '[:upper:]')
  TOKEN_NAME="${UPPER_NAME}_SAS_TOKEN"
  CONTAINER_NAME="AZURE_${UPPER_NAME}_CONTAINER_NAME"
  KEY_KID="$(echo "$TDP" | sed -E 's/(^|_)([a-z])/\U\2/g')FilesystemEncryptionKey"
  cat >> "$AZ_SCRIPT_7" <<EOF
TMP=\`echo \$TMP | \\
  jq '.azure_filesystems[$idx].azure_url = "https://" + env.AZURE_STORAGE_ACCOUNT_NAME + ".blob.core.windows.net/" + env.${CONTAINER_NAME} + "/data.img" + env.${TOKEN_NAME}' | \\
  jq '.azure_filesystems[$idx].mount_point = "/mnt/remote/${TDP}"' | \\
  jq '.azure_filesystems[$idx].key.kid = "${KEY_KID}"' | \\
  jq '.azure_filesystems[$idx].key.kty = env.AZURE_AKV_KEY_TYPE' | \\
  jq '.azure_filesystems[$idx].key.akv.endpoint = env.AZURE_KEYVAULT_ENDPOINT' | \\
  jq '.azure_filesystems[$idx].key.akv.bearer_token = env.BEARER_TOKEN' | \\
  jq '.azure_filesystems[$idx].key_derivation.label = "${KEY_KID}"' | \\
  jq '.azure_filesystems[$idx].key_derivation.salt = "9b53cddbe5b78a0b912a8f05f341bcd4dd839ea85d26a08efaef13e696d999f4"'\`

EOF
  idx=$((idx+1))
done

if [ "$MODEL_FORMAT" != "ccr_instantiate" ]; then
    cat >> "$AZ_SCRIPT_7" <<EOF
TMP=\`echo \$TMP | \\
  jq '.azure_filesystems[$idx].azure_url = "https://" + env.AZURE_STORAGE_ACCOUNT_NAME + ".blob.core.windows.net/" + env.AZURE_MODEL_CONTAINER_NAME + "/data.img" + env.MODEL_SAS_TOKEN' | \\
  jq '.azure_filesystems[$idx].mount_point = "/mnt/remote/model"' | \\
  jq '.azure_filesystems[$idx].key.kid = "ModelFilesystemEncryptionKey"' | \\
  jq '.azure_filesystems[$idx].key.kty = env.AZURE_AKV_KEY_TYPE' | \\
  jq '.azure_filesystems[$idx].key.akv.endpoint = env.AZURE_KEYVAULT_ENDPOINT' | \\
  jq '.azure_filesystems[$idx].key.akv.bearer_token = env.BEARER_TOKEN' | \\
  jq '.azure_filesystems[$idx].key_derivation.label = "ModelFilesystemEncryptionKey"' | \\
  jq '.azure_filesystems[$idx].key_derivation.salt = "9b53cddbe5b78a0b912a8f05f341bcd4dd839ea85d26a08efaef13e696d999f4"'\`

EOF
  idx=$((idx+1))
fi

cat >> "$AZ_SCRIPT_7" <<EOF
TMP=\`echo \$TMP | \\
  jq '.azure_filesystems[$idx].azure_url = "https://" + env.AZURE_STORAGE_ACCOUNT_NAME + ".blob.core.windows.net/" + env.AZURE_OUTPUT_CONTAINER_NAME + "/data.img" + env.OUTPUT_SAS_TOKEN' | \\
  jq '.azure_filesystems[$idx].mount_point = "/mnt/remote/output"' | \\
  jq '.azure_filesystems[$idx].key.kid = "OutputFilesystemEncryptionKey"' | \\
  jq '.azure_filesystems[$idx].key.kty = env.AZURE_AKV_KEY_TYPE' | \\
  jq '.azure_filesystems[$idx].key.akv.endpoint = env.AZURE_KEYVAULT_ENDPOINT' | \\
  jq '.azure_filesystems[$idx].key.akv.bearer_token = env.BEARER_TOKEN' | \\
  jq '.azure_filesystems[$idx].key_derivation.label = "OutputFilesystemEncryptionKey"' | \\
  jq '.azure_filesystems[$idx].key_derivation.salt = "9b53cddbe5b78a0b912a8f05f341bcd4dd839ea85d26a08efaef13e696d999f4"'\`

EOF

cat >> "$AZ_SCRIPT_7" <<'EOF'
ENCRYPTED_FILESYSTEM_INFORMATION=`echo $TMP | base64 --wrap=0`
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
EOF

chmod +x "$AZ_SCRIPT_7"


# generatefs.sh

AZ_SCRIPT_8="$SCENARIO_DIR/deployment/azure/generatefs.sh"
cat > "$AZ_SCRIPT_8" <<'EOF'
#!/bin/bash

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

while getopts ":d:k:i:" options; do
    case $options in 
        d)dataPath=$OPTARG;;
        k)keyFilePath=$OPTARG;;
        i)encryptedImage=$OPTARG;;
    esac
done

echo Encrypting $dataPath with key $keyFilePath and generating $encryptedImage
deviceName=cryptdevice1
deviceNamePath="/dev/mapper/$deviceName"

if [ -f "$keyFilePath" ]; then
    echo "[!] Encrypting dataset using $keyFilePath"
else
    echo "[!] Generating keyfile..."
    dd if=/dev/random of="$keyFilePath" count=1 bs=32
    truncate -s 32 "$keyFilePath"
fi

echo "[!] Creating encrypted image..."

response=`du -s $dataPath`
read -ra arr <<< "$response"
size=`echo "x=l($arr)/l(2); scale=0; 2^((x+0.5)/1)*2" | bc -l;`

# cryptsetup requires 16M or more

if (($((size)) < 65536)); then 
    size="65536"
fi
size=$size"K"

echo "Data size: $size"

rm -f "$encryptedImage"
touch "$encryptedImage"
truncate --size $size "$encryptedImage"

sudo cryptsetup luksFormat --type luks2 "$encryptedImage" \
    --key-file "$keyFilePath" -v --batch-mode --sector-size 4096 \
    --cipher aes-xts-plain64 \
    --pbkdf pbkdf2 --pbkdf-force-iterations 1000

sudo cryptsetup luksOpen "$encryptedImage" "$deviceName" \
    --key-file "$keyFilePath" \
    --integrity-no-journal --persistent

echo "[!] Formatting as ext4..."

sudo mkfs.ext4 "$deviceNamePath"

echo "[!] Mounting..."

mountPoint=`mktemp -d`
echo "Mounting to $mountPoint"
sudo mount -t ext4 "$deviceNamePath" "$mountPoint" -o loop

echo "[!] Copying contents to encrypted device..."

# The /* is needed to copy folder contents instead of the folder + contents
sudo cp -r $dataPath/* "$mountPoint"
sudo rm -rf "$mountPoint/lost+found"
ls "$mountPoint"

echo "[!] Closing device..."

sudo umount "$mountPoint"
sleep 2
sudo cryptsetup luksClose "$deviceName"
EOF

chmod +x "$AZ_SCRIPT_8"


# aci-parameters-template.json

AZ_SCRIPT_9="$SCENARIO_DIR/deployment/azure/aci-parameters-template.json"
jq -n \
'
{
  "containerRegistry": {
    "value": ""
  },
  "ccePolicy": {
    "value": ""
  },
  "EncfsSideCarArgs": {
    "value": ""
  },
  "ContractService": {
    "value": ""
  },
  "ContractServiceParameters": {
    "value": ""
  },
  "Contracts": {
    "value": ""
  },
  "PipelineConfiguration": {
    "value": ""
  }
}
' > "$AZ_SCRIPT_9"


# arm-template.json

AZ_SCRIPT_10="$SCENARIO_DIR/deployment/azure/arm-template.json"
jq -n \
  --arg scenarioName "depa-training-$SCENARIO_NAME" \
  '
  {
    "$schema": "https://schema.management.azure.com/schemas/2019-04-01/deploymentTemplate.json#",
    "contentVersion": "1.0.0.0",
    "parameters": {
        "name": {
        "defaultValue": $scenarioName,
        "type": "string",
        "metadata": {
            "description": "Name for the container group"
        }
        },
        "location": {
        "defaultValue": "northeurope",
        "type": "string",
        "metadata": {
            "description": "Location for all resources."
        }
        },
        "port": {
        "defaultValue": 8080,
        "type": "int",
        "metadata": {
            "description": "Port to open on the container and the public IP address."
        }
        },
        "containerRegistry": {
        "defaultValue": "secureString",
        "type": "string",
        "metadata": {
            "description": "The container registry login server."
        }
        },
        "restartPolicy": {
        "defaultValue": "Never",
        "allowedValues": [
            "Always",
            "Never",
            "OnFailure"
        ],
        "type": "string",
        "metadata": {
            "description": "The behavior of Azure runtime if container has stopped."
        }
        },
        "ccePolicy": {
        "defaultValue": "secureString",
        "type": "string",
        "metadata": {
            "description": "cce policy"
        }
        },
        "EncfsSideCarArgs": {
        "defaultValue": "secureString",
        "type": "string",
        "metadata": {
            "description": "Remote file system information for storage sidecar."
        }
        },
        "ContractService": {
        "defaultValue": "secureString",
        "type": "string",
        "metadata": {
            "description": "URL of contract service"
        }
        },
        "Contracts": {
        "defaultValue": "secureString",
        "type": "string",
        "metadata": {
            "description": "List of contracts"
        }
        },
        "ContractServiceParameters": {
        "defaultValue": "secureString",
        "type": "string",
        "metadata": {
            "description": "Contract service parameters"
        }
        },
        "PipelineConfiguration": {
        "defaultValue": "secureString",
        "type": "string",
        "metadata": {
            "description": "Pipeline configuration"
        }
        }
    },
    "resources": [
        {
        "type": "Microsoft.ContainerInstance/containerGroups",
        "apiVersion": "2023-05-01",
        "name": "[parameters('name')]",
        "location": "[parameters('location')]",
        "properties": {
            "confidentialComputeProperties": {
            "ccePolicy": "[parameters('ccePolicy')]"
            },
            "containers": [
            {
                "name": "depa-training",
                "properties": {
                "image": "[concat(parameters('containerRegistry'), '/depa-training:latest')]",
                "command": [
                    "/bin/bash",
                    "run.sh"
                ],
                "environmentVariables": [],
                "volumeMounts": [
                    {
                    "name": "remotemounts",
                    "mountPath": "/mnt/remote"
                    }
                ],
                "resources": {
                    "requests": {
                    "cpu": 3,
                    "memoryInGB": 12
                    }
                }
                }
            },
            {
                "name": "encrypted-storage-sidecar",
                "properties": {
                "image": "[concat(parameters('containerRegistry'), '/depa-training-encfs:latest')]",
                "command": [
                    "/encfs.sh"
                ],
                "environmentVariables": [
                    {
                    "name": "EncfsSideCarArgs",
                    "value": "[parameters('EncfsSideCarArgs')]"
                    },
                    {
                    "name": "ContractService",
                    "value": "[parameters('ContractService')]"
                    },
                    {
                    "name": "Contracts",
                    "value": "[parameters('Contracts')]"
                    },
                    {
                    "name": "ContractServiceParameters",
                    "value": "[parameters('ContractServiceParameters')]"
                    },
                    {
                    "name": "PipelineConfiguration",
                    "value": "[parameters('PipelineConfiguration')]"
                    }
                ],
                "volumeMounts": [
                    {
                    "name": "remotemounts",
                    "mountPath": "/mnt/remote"
                    }
                ],
                "securityContext": {
                    "privileged": "true"
                },
                "resources": {
                    "requests": {
                    "cpu": 0.5,
                    "memoryInGB": 2
                    }
                }
                }
            }
            ],
            "sku": "Confidential",
            "osType": "Linux",
            "restartPolicy": "[parameters('restartPolicy')]",
            "volumes": [
            {
                "name": "remotemounts",
                "emptydir": {}
            }
            ]
        }
        }
    ]
  }
  ' > "$AZ_SCRIPT_10"


# encrypted-filesystem-config-template.json

AZ_SCRIPT_11="$SCENARIO_DIR/deployment/azure/encrypted-filesystem-config-template.json"

# Initialize the file with empty azure_filesystems array
jq -n '{"azure_filesystems": []}' > "$AZ_SCRIPT_11"

# Add one filesystem for each TDP
for TDP in "${TDP_NAMES[@]}"; do
  jq --argjson new_fs '{
    "azure_url": "",
    "azure_url_private": false,
    "read_write": false,
    "mount_point": "",
    "key": {
      "kid": "",
      "kty": "",
      "authority": {
        "endpoint": "sharedneu.neu.attest.azure.net"
      },
      "akv": {
        "endpoint": "",
        "api_version": "api-version=7.3-preview",
        "bearer_token": ""
      }
    },
    "key_derivation": {
      "salt": "",
      "label": ""
    }
  }' '.azure_filesystems += [$new_fs]' "$AZ_SCRIPT_11" > "$AZ_SCRIPT_11.tmp" && mv "$AZ_SCRIPT_11.tmp" "$AZ_SCRIPT_11"
done

# Add one filesystem for the output container with read_write true
jq --argjson new_fs '{
  "azure_url": "",
  "azure_url_private": false,
  "read_write": true,
  "mount_point": "",
  "key": {
    "kid": "",
    "kty": "",
    "authority": {
      "endpoint": "sharedneu.neu.attest.azure.net"
    },
    "akv": {
      "endpoint": "",
      "api_version": "api-version=7.3-preview",
      "bearer_token": ""
    }
  },
  "key_derivation": {
    "salt": "",
    "label": ""
  }
}' '.azure_filesystems += [$new_fs]' "$AZ_SCRIPT_11" > "$AZ_SCRIPT_11.tmp" && mv "$AZ_SCRIPT_11.tmp" "$AZ_SCRIPT_11"


######## Create deployment/local ########

# docker-compose-preprocess.yml

PREPROCESS_YML="$SCENARIO_DIR/deployment/local/docker-compose-preprocess.yml"

echo "services:" > "$PREPROCESS_YML"
for TDP in "${TDP_NAMES[@]}"; do
  # replace _ with - and make lowercase
  TDP_SUFFIX=$(echo "$TDP" | tr '_' '-' | tr '[:upper:]' '[:lower:]')
  UPPER_TDP=$(echo "$TDP" | tr '[:lower:]' '[:upper:]')
  TDP_INPUT_PATH="\$${UPPER_TDP}_INPUT_PATH"
  TDP_OUTPUT_PATH="\$${UPPER_TDP}_OUTPUT_PATH"
  echo "  $TDP:" >> "$PREPROCESS_YML"
  echo "    image: \${CONTAINER_REGISTRY:+\$CONTAINER_REGISTRY/}preprocess-${TDP_SUFFIX}:latest" >> "$PREPROCESS_YML"
  echo "    volumes:" >> "$PREPROCESS_YML"
  echo "      - $TDP_INPUT_PATH:/mnt/input/data" >> "$PREPROCESS_YML"
  echo "      - $TDP_OUTPUT_PATH:/mnt/output/preprocessed" >> "$PREPROCESS_YML"
  echo "    command: ["python3", "preprocess_${TDP}.py"]" >> "$PREPROCESS_YML"
done

# preprocess.sh

PREPROCESS_SH="$SCENARIO_DIR/deployment/local/preprocess.sh"

echo "#!/bin/bash" > "$PREPROCESS_SH"
echo "export REPO_ROOT=\"\$(git rev-parse --show-toplevel)\"" >> "$PREPROCESS_SH"
echo "export SCENARIO=\"$SCENARIO_NAME\"" >> "$PREPROCESS_SH"
echo "export DATA_DIR=\$REPO_ROOT/scenarios/\$SCENARIO/data" >> "$PREPROCESS_SH"
for TDP in "${TDP_NAMES[@]}"; do
  UPPER_TDP=$(echo "$TDP" | tr '[:lower:]' '[:upper:]')
  echo "export ${UPPER_TDP}_INPUT_PATH=\$DATA_DIR/$TDP" >> "$PREPROCESS_SH"
  echo "export ${UPPER_TDP}_OUTPUT_PATH=\$DATA_DIR/$TDP/preprocessed" >> "$PREPROCESS_SH"
done
for TDP in "${TDP_NAMES[@]}"; do
  UPPER_TDP=$(echo "$TDP" | tr '[:lower:]' '[:upper:]')
  echo "mkdir -p \$${UPPER_TDP}_OUTPUT_PATH" >> "$PREPROCESS_SH"
done
echo "docker compose -f docker-compose-preprocess.yml up --remove-orphans" >> "$PREPROCESS_SH"

chmod +x "$PREPROCESS_SH"

# docker-compose-modelsave.yml & save-model.sh

if [ "$MODEL_FORMAT" != "ccr_instantiate" ]; then
  MODEL_YML="$SCENARIO_DIR/deployment/local/docker-compose-modelsave.yml"
  SCENARIO_PREFIX=$(echo "$SCENARIO_NAME" | tr '_' '-')
  echo "services:" > "$MODEL_YML"
  echo "  model_save:" >> "$MODEL_YML"
  echo "    image: \${CONTAINER_REGISTRY:+\$CONTAINER_REGISTRY/}${SCENARIO_PREFIX}-model-save:latest" >> "$MODEL_YML"
  echo "    volumes:" >> "$MODEL_YML"
  echo "      - \$MODEL_OUTPUT_PATH:/mnt/model" >> "$MODEL_YML"
  echo "      - \$MODEL_CONFIG_PATH:/mnt/config/model_config.json" >> "$MODEL_YML"
  echo "    command: ["python3", "save_base_model.py"]" >> "$MODEL_YML"
  chmod +x "$MODEL_YML"

  MODEL_SAVE_SH="$SCENARIO_DIR/deployment/local/save-model.sh"
  echo "#!/bin/bash" > "$MODEL_SAVE_SH"
  echo "export REPO_ROOT=\"\$(git rev-parse --show-toplevel)\"" >> "$MODEL_SAVE_SH"
  echo "export SCENARIO=\"$SCENARIO_NAME\"" >> "$MODEL_SAVE_SH"
  echo "export MODEL_OUTPUT_PATH=\$REPO_ROOT/scenarios/\$SCENARIO/modeller/models" >> "$MODEL_SAVE_SH"
  echo "mkdir -p \$MODEL_OUTPUT_PATH" >> "$MODEL_SAVE_SH"
  echo "export MODEL_CONFIG_PATH=\$REPO_ROOT/scenarios/\$SCENARIO/config/model_config.json" >> "$MODEL_SAVE_SH"
  echo "docker compose -f docker-compose-modelsave.yml up --remove-orphans" >> "$MODEL_SAVE_SH"
  chmod +x "$MODEL_SAVE_SH"
fi

# docker-compose-train.yml

TRAIN_YML="$SCENARIO_DIR/deployment/local/docker-compose-train.yml"

echo "services:" > "$TRAIN_YML"
echo "  train:" >> "$TRAIN_YML"
echo "    image: \${CONTAINER_REGISTRY:+\$CONTAINER_REGISTRY/}depa-training:latest" >> "$TRAIN_YML"
echo "    volumes:" >> "$TRAIN_YML"
for TDP in "${TDP_NAMES[@]}"; do
  UPPER_TDP=$(echo "$TDP" | tr '[:lower:]' '[:upper:]')
  echo "      - \$${UPPER_TDP}_INPUT_PATH:/mnt/remote/${TDP}" >> "$TRAIN_YML"
done
if [ "$MODEL_FORMAT" != "ccr_instantiate" ]; then
  echo "      - \$MODEL_INPUT_PATH:/mnt/remote/model" >> "$TRAIN_YML"
fi
echo "      - \$MODEL_OUTPUT_PATH:/mnt/remote/output" >> "$TRAIN_YML"
echo "      - \$CONFIGURATION_PATH:/mnt/remote/config" >> "$TRAIN_YML"
echo "    command: ["/bin/bash", "run.sh"]" >> "$TRAIN_YML"

# train.sh

TRAIN_SH="$SCENARIO_DIR/deployment/local/train.sh"
echo "#!/bin/bash" > "$TRAIN_SH"
echo "export REPO_ROOT=\"\$(git rev-parse --show-toplevel)\"" >> "$TRAIN_SH"
echo "export SCENARIO=\"$SCENARIO_NAME\"" >> "$TRAIN_SH"
echo "export DATA_DIR=\$REPO_ROOT/scenarios/\$SCENARIO/data" >> "$TRAIN_SH"
echo "export MODEL_DIR=\$REPO_ROOT/scenarios/\$SCENARIO/modeller" >> "$TRAIN_SH"
for TDP in "${TDP_NAMES[@]}"; do
  UPPER_TDP=$(echo "$TDP" | tr '[:lower:]' '[:upper:]')
  echo "export \$${UPPER_TDP}_INPUT_PATH=\$DATA_DIR/${TDP}/preprocessed" >> "$TRAIN_SH"
done
if [ "$MODEL_FORMAT" != "ccr_instantiate" ]; then
  echo "export MODEL_INPUT_PATH=\$MODEL_DIR/models" >> "$TRAIN_SH"
fi
echo "export MODEL_OUTPUT_PATH=\$MODEL_DIR/output" >> "$TRAIN_SH"
echo "sudo rm -rf \$MODEL_OUTPUT_PATH" >> "$TRAIN_SH"
echo "mkdir -p \$MODEL_OUTPUT_PATH" >> "$TRAIN_SH"
echo "export CONFIGURATION_PATH=\$REPO_ROOT/scenarios/\$SCENARIO/config" >> "$TRAIN_SH"
echo "\$REPO_ROOT/scenarios/\$SCENARIO/config/consolidate_pipeline.sh" >> "$TRAIN_SH"
echo "docker compose -f docker-compose-train.yml up --remove-orphans" >> "$TRAIN_SH"
chmod +x "$TRAIN_SH"

######## Create config/templates ########

# consolidate_pipeline.sh

CONSOLIDATE_PIPELINE_SH="$SCENARIO_DIR/config/consolidate_pipeline.sh"
cat << EOF > "$CONSOLIDATE_PIPELINE_SH"
#! /bin/bash

REPO_ROOT="\$(git rev-parse --show-toplevel)"
SCENARIO="$SCENARIO_NAME"

template_path="\$REPO_ROOT/scenarios/\$SCENARIO/config/templates"
model_config_path="\$REPO_ROOT/scenarios/\$SCENARIO/config/model_config.json"
data_config_path="\$REPO_ROOT/scenarios/\$SCENARIO/config/dataset_config.json"
loss_config_path="\$REPO_ROOT/scenarios/\$SCENARIO/config/loss_config.json"
train_config_path="\$REPO_ROOT/scenarios/\$SCENARIO/config/train_config.json"
eval_config_path="\$REPO_ROOT/scenarios/\$SCENARIO/config/eval_config.json"
join_config_path="\$REPO_ROOT/scenarios/\$SCENARIO/config/join_config.json"
pipeline_config_path="\$REPO_ROOT/scenarios/\$SCENARIO/config/pipeline_config.json"

# populate "model_config", "data_config", and "loss_config" keys in train config
train_config=\$(cat \$template_path/train_config_template.json)

# Only merge if the file exists
if [[ -f "\$model_config_path" ]]; then
    model_config=\$(cat \$model_config_path)
    train_config=\$(echo "\$train_config" | jq --argjson model "\$model_config" '.config.model_config = \$model')
fi

if [[ -f "\$data_config_path" ]]; then
    data_config=\$(cat \$data_config_path)
    train_config=\$(echo "\$train_config" | jq --argjson data "\$data_config" '.config.dataset_config = \$data')
fi

if [[ -f "\$loss_config_path" ]]; then
    loss_config=\$(cat \$loss_config_path)
    train_config=\$(echo "\$train_config" | jq --argjson loss "\$loss_config" '.config.loss_config = \$loss')
fi

if [[ -f "\$eval_config_path" ]]; then
    eval_config=\$(cat \$eval_config_path)
    # Get all keys from eval_config and copy them to train_config
    for key in \$(echo "\$eval_config" | jq -r 'keys[]'); do
        train_config=\$(echo "\$train_config" | jq --argjson eval "\$eval_config" --arg key "\$key" '.config[\$key] = \$eval[\$key]')
    done
fi

# save train_config
echo "\$train_config" > \$train_config_path

# prepare pipeline config from join_config.json (first dict "config") and train_config.json (second dict "config")
pipeline_config=\$(cat \$template_path/pipeline_config_template.json)

# Only merge join_config if the file exists
if [[ -f "\$join_config_path" ]]; then
    join_config=\$(cat \$join_config_path)
    pipeline_config=\$(echo "\$pipeline_config" | jq --argjson join "\$join_config" '.pipeline += [\$join]')
fi

# Always merge train_config as it's required
pipeline_config=\$(echo "\$pipeline_config" | jq --argjson train "\$train_config" '.pipeline += [\$train]')

# save pipeline_config to pipeline_config.json
echo "\$pipeline_config" > \$pipeline_config_path
EOF

chmod +x "$CONSOLIDATE_PIPELINE_SH"

# templates/pipeline_config_template.json

PIPELINE_CONFIG_TEMPLATE="$SCENARIO_DIR/config/templates/pipeline_config_template.json"
cat << EOF > "$PIPELINE_CONFIG_TEMPLATE"
{
  "pipeline": []
}
EOF
TRAIN_CONFIG_TEMPLATE="$SCENARIO_DIR/config/templates/train_config_template.json"
PRIVACY="false"
# Check if any dataset has privacy enabled
if jq -e '.tdps[].datasets[] | select(.privacy == true)' "$INPUT_JSON" > /dev/null 2>&1; then
  PRIVACY="true"
fi

# Get minimum epsilon and delta
if [ "$PRIVACY" = "true" ]; then
  MIN_EPSILON=$(jq -r '[.tdps[].datasets[] | select(.epsilon != null) | .epsilon] | min' "$INPUT_JSON")
  MIN_DELTA=$(jq -r '[.tdps[].datasets[] | select(.delta != null) | .delta] | min' "$INPUT_JSON")
  if [ "$MIN_DELTA" != "null" ]; then
    MECHANISM="gaussian"
  else
    MECHANISM="laplace"
  fi
else
  MIN_EPSILON="null"
  MIN_DELTA="null"
  MECHANISM="null"
fi

# templates/train_config_template.json

cat << EOF > "$TRAIN_CONFIG_TEMPLATE"
{
  "name": "$TRAIN_METHOD",
  "config": {
    "paths": {
      "input_dataset_path": "/tmp/",
      "trained_model_output_path": "/mnt/remote/output"
    },
    "is_private": $PRIVACY,
    "privacy_params": {
      "epsilon": $MIN_EPSILON,
      "delta": $MIN_DELTA,
      "mechanism": "$MECHANISM"
    }
  }
}
EOF

# If join_type exists in scenario config, create a join_config.json
if [ -n "$JOIN_TYPE" ]; then
  JOIN_CONFIG="$SCENARIO_DIR/config/join_config.json"
  
  # Initialize the config with empty datasets array
  jq -n "{\"name\": \"$JOIN_TYPE\", \"config\": {\"datasets\": []}}" > "$JOIN_CONFIG"
  
  for TDP in "${TDP_NAMES[@]}"; do
    # Get datasets for this TDP and iterate over them properly
    jq -r --arg tdp "$TDP" '.tdps[] | select(.name == $tdp) | .datasets[] | @json' "$INPUT_JSON" | while read -r dataset; do
      # Extract fields from the dataset JSON
      dataset_id=$(echo "$dataset" | jq -r '.id')
      dataset_name=$(echo "$dataset" | jq -r '.name')
      # Create the dataset object
      dataset_obj=$(jq -n "{
        \"id\": \"$dataset_id\",
        \"provider\": \"$TDP\",
        \"name\": \"$dataset_name\",
        \"file\": \"\",
        \"select_variables\": [],
        \"mount_path\": \"/mnt/remote/$TDP/\"
      }")
      # Append to the datasets array
      jq --argjson new_dataset "$dataset_obj" '.config.datasets += [$new_dataset]' "$JOIN_CONFIG" > "$JOIN_CONFIG.tmp" && mv "$JOIN_CONFIG.tmp" "$JOIN_CONFIG"
    done
  done
  
  # Add the joined_dataset to config
  if [ "$JOIN_TYPE" = "SparkJoin" ]; then
    joined_dataset_obj=$(jq -n '{
      "joined_dataset": "/tmp/",
      "joining_query": "",
      "joining_key": "",
      "drop_columns": [],
      "identifiers": []
    }')
    jq --argjson joined_dataset "$joined_dataset_obj" '.config.joined_dataset = $joined_dataset' "$JOIN_CONFIG" > "$JOIN_CONFIG.tmp" && mv "$JOIN_CONFIG.tmp" "$JOIN_CONFIG"
  fi
  if [ "$JOIN_TYPE" = "DirectoryJoin" ]; then
    jq '.config.joined_dataset = "/tmp/"' "$JOIN_CONFIG" > "$JOIN_CONFIG.tmp" && mv "$JOIN_CONFIG.tmp" "$JOIN_CONFIG"
  fi
fi

# templates/eval_config_template.json

EVAL_CONFIG="$SCENARIO_DIR/config/eval_config.json"
cat << EOF > "$EVAL_CONFIG"
{
  "task_type": "",
  "metrics": []
}
EOF

# dataset_config.json

DATASET_CONFIG="$SCENARIO_DIR/config/dataset_config.json"
cat << EOF > "$DATASET_CONFIG"
{
  "type": "",
  "target_variable": "",
  "missing_strategy": "",
  "splits": {},
  "data_type": ""
}
EOF

if [ "$MODEL_FORMAT" != "ONNX" ]; then
  MODEL_CONFIG="$SCENARIO_DIR/config/model_config.json"
  if [ "$TRAIN_METHOD" = "Train_XGB" ]; then
    cat << EOF > "$MODEL_CONFIG"
{
  "num_boost_round": 0,
  "booster_params": {
    "max_depth": 0,
    "learning_rate": 0.0,
    "objective": "",
    "eval_metric": ""
  }
}
EOF
  fi

  if [ "$TRAIN_METHOD" = "Train_DL" ]; then
  cat << EOF > "$MODEL_CONFIG"
{
  "submodules": "",
  "layers": "",
  "input": "",
  "forward": "",
  "output": ""
}
EOF
  
  # loss_config.json
  LOSS_CONFIG="$SCENARIO_DIR/config/loss_config.json"
  cat << EOF > "$LOSS_CONFIG"
{
  "expression": "",
  "components": {},
  "variables": {},
  "reduction": ""
}
EOF
  fi
fi


######## Create export-variables.sh ########

EXPORT_VARIABLES_SH="$SCENARIO_DIR/export-variables.sh"
cat << EOF > "$EXPORT_VARIABLES_SH"
#!/bin/bash

# Azure Naming Rules:
#
# Resource Group:
# - 1-90 characters
# - Letters, numbers, underscores, parentheses, hyphens, periods allowed
# - Cannot end with a period (.)
# - Case-insensitive, unique within subscription
#
# Key Vault:
# - 3-24 characters
# - Globally unique name
# - Lowercase letters, numbers, hyphens only
# - Must start and end with letter or number
#
# Storage Account:
# - 3-24 characters
# - Globally unique name
# - Lowercase letters and numbers only
#
# Storage Container:
# - 3-63 characters
# - Lowercase letters, numbers, hyphens only
# - Must start and end with a letter or number
# - No consecutive hyphens
# - Unique within storage account

EOF

for TDP in "${TDP_NAMES[@]}"; do
  UPPER_TDP=$(echo "$TDP" | tr '-' '_' | tr '[:lower:]' '[:upper:]')
  LOWER_TDP=$(echo "$TDP" | sed 's/[_|-]//g' | tr '[:upper:]' '[:lower:]')
  echo "declare -x AZURE_${UPPER_TDP}_CONTAINER_NAME=${LOWER_TDP}container" >> "$EXPORT_VARIABLES_SH"
  echo "export AZURE_${UPPER_TDP}_CONTAINER_NAME" >> "$EXPORT_VARIABLES_SH"
done

if [ "$MODEL_FORMAT" != "ccr_instantiate" ]; then
  echo "declare -x AZURE_MODEL_CONTAINER_NAME=modelcontainer" >> "$EXPORT_VARIABLES_SH"
  echo "export AZURE_MODEL_CONTAINER_NAME" >> "$EXPORT_VARIABLES_SH"
fi
echo "declare -x AZURE_OUTPUT_CONTAINER_NAME=outputcontainer" >> "$EXPORT_VARIABLES_SH"
echo "export AZURE_OUTPUT_CONTAINER_NAME" >> "$EXPORT_VARIABLES_SH"

cat >> "$EXPORT_VARIABLES_SH" <<EOF

# For cloud resource creation:
declare -x SCENARIO="$SCENARIO_NAME"
declare -x REPO_ROOT="\$(git rev-parse --show-toplevel)"
declare -x CONTAINER_REGISTRY=ispirt.azurecr.io
declare -x AZURE_LOCATION=<azure-location>
declare -x AZURE_SUBSCRIPTION_ID=
declare -x AZURE_RESOURCE_GROUP=
declare -x AZURE_KEYVAULT_ENDPOINT=
declare -x AZURE_STORAGE_ACCOUNT_NAME=

# For key import:
declare -x CONTRACT_SERVICE_URL=https://<contract-service-url>:8000
declare -x TOOLS_HOME=\$REPO_ROOT/external/confidential-sidecar-containers/tools

# Export all variables to make them available to other scripts
export SCENARIO
export REPO_ROOT
export CONTAINER_REGISTRY
export AZURE_LOCATION
export AZURE_SUBSCRIPTION_ID
export AZURE_RESOURCE_GROUP
export AZURE_KEYVAULT_ENDPOINT
export AZURE_STORAGE_ACCOUNT_NAME
export CONTRACT_SERVICE_URL
export TOOLS_HOME
EOF

chmod +x "$EXPORT_VARIABLES_SH"

# .gitignore

GITIGNORE="$SCENARIO_DIR/.gitignore"
cat << EOF > "$GITIGNORE"
data/
modeller/
venv/

*.img
*.bin
*.csv
*.parquet
*.onnx
*.gz
*.safetensors

**/__pycache__/
EOF