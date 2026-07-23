#!/bin/bash

set -euo pipefail

required_vars=(
  BANK_1_TDP_DID
  BANK_2_TDP_DID
  BANK_3_TDP_DID
  TDC_DID
  CCRP_DID
  AZURE_KEYVAULT_ENDPOINT
  AZURE_STORAGE_ACCOUNT_NAME
  AZURE_BANK_1_CONTAINER_NAME
  AZURE_BANK_2_CONTAINER_NAME
  AZURE_BANK_3_CONTAINER_NAME
)

for required_var in "${required_vars[@]}"; do
  if [[ -z "${!required_var:-}" ]]; then
    echo "Environment variable $required_var not defined" >&2
    exit 1
  fi
done

validate_web_did() {
  local variable_name="$1"
  local did="$2"
  if [[ ! "$did" =~ ^did:web:[^[:space:]]+$ ]]; then
    echo "$variable_name must be a did:web identifier" >&2
    exit 1
  fi
}

validate_web_did BANK_1_TDP_DID "$BANK_1_TDP_DID"
validate_web_did BANK_2_TDP_DID "$BANK_2_TDP_DID"
validate_web_did BANK_3_TDP_DID "$BANK_3_TDP_DID"
validate_web_did TDC_DID "$TDC_DID"
validate_web_did CCRP_DID "$CCRP_DID"

if [[ "$BANK_1_TDP_DID" == "$BANK_2_TDP_DID" || \
      "$BANK_1_TDP_DID" == "$BANK_3_TDP_DID" || \
      "$BANK_2_TDP_DID" == "$BANK_3_TDP_DID" ]]; then
  echo "The three bank TDP DIDs must be distinct" >&2
  exit 1
fi

if [[ "$AZURE_KEYVAULT_ENDPOINT" != https://*.vault.azure.net && \
      "$AZURE_KEYVAULT_ENDPOINT" != https://*.managedhsm.azure.net ]]; then
  echo "AZURE_KEYVAULT_ENDPOINT must be an Azure Key Vault or Managed HSM HTTPS endpoint" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_FILE="${1:-/tmp/mule-detection-contract.json}"

START_TIME="$(date -u '+%Y-%m-%dT%H:%M:%S.000Z')"
EXPIRY_TIME="$(date -u -d '+365 days' '+%Y-%m-%dT%H:%M:%S.000Z')"
CONTRACT_ID="${CONTRACT_ID:-$(cat /proc/sys/kernel/random/uuid)}"

BANK_1_URL="https://${AZURE_STORAGE_ACCOUNT_NAME}.blob.core.windows.net/${AZURE_BANK_1_CONTAINER_NAME}/data.img"
BANK_2_URL="https://${AZURE_STORAGE_ACCOUNT_NAME}.blob.core.windows.net/${AZURE_BANK_2_CONTAINER_NAME}/data.img"
BANK_3_URL="https://${AZURE_STORAGE_ACCOUNT_NAME}.blob.core.windows.net/${AZURE_BANK_3_CONTAINER_NAME}/data.img"

jq \
  --arg id "$CONTRACT_ID" \
  --arg bank_1_tdp "$BANK_1_TDP_DID" \
  --arg bank_2_tdp "$BANK_2_TDP_DID" \
  --arg bank_3_tdp "$BANK_3_TDP_DID" \
  --arg tdc "$TDC_DID" \
  --arg ccrp "$CCRP_DID" \
  --arg endpoint "$AZURE_KEYVAULT_ENDPOINT" \
  --arg start_time "$START_TIME" \
  --arg expiry_time "$EXPIRY_TIME" \
  --arg bank_1_url "$BANK_1_URL" \
  --arg bank_2_url "$BANK_2_URL" \
  --arg bank_3_url "$BANK_3_URL" \
  '.id = $id
   | .tdc = $tdc
   | .tdps = [$bank_1_tdp, $bank_2_tdp, $bank_3_tdp]
   | .ccrp = $ccrp
   | .startTime = $start_time
   | .expiryTime = $expiry_time
   | .datasets |= map(
       .key.properties.endpoint = $endpoint
       | if .owner == "bank_1" then .url = $bank_1_url | .provider = $bank_1_tdp
         elif .owner == "bank_2" then .url = $bank_2_url | .provider = $bank_2_tdp
         elif .owner == "bank_3" then .url = $bank_3_url | .provider = $bank_3_tdp
         else .
         end
     )' \
  "$SCRIPT_DIR/contract.json" > "$OUTPUT_FILE"

if grep -Eq '\$[A-Z_]+|<[^>]+>' "$OUTPUT_FILE"; then
  echo "Prepared contract still contains unresolved placeholders" >&2
  rm -f "$OUTPUT_FILE"
  exit 1
fi

echo "Prepared unsigned contract for the configured participants: $OUTPUT_FILE"
echo "Sign and register it with the official iSPIRT/contract-service flow."
