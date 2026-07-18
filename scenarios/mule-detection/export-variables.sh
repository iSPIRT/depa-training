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

declare -x AZURE_BANK_1_CONTAINER_NAME=bank1container
export AZURE_BANK_1_CONTAINER_NAME
declare -x AZURE_BANK_2_CONTAINER_NAME=bank2container
export AZURE_BANK_2_CONTAINER_NAME
declare -x AZURE_BANK_3_CONTAINER_NAME=bank3container
export AZURE_BANK_3_CONTAINER_NAME
declare -x AZURE_OUTPUT_CONTAINER_NAME=outputcontainer
export AZURE_OUTPUT_CONTAINER_NAME

# For cloud resource creation:
declare -x SCENARIO="mule-detection"
declare -x REPO_ROOT="$(git rev-parse --show-toplevel)"
declare -x CONTAINER_REGISTRY=ispirt.azurecr.io
declare -x AZURE_LOCATION=northeurope
declare -x AZURE_SUBSCRIPTION_ID=
declare -x AZURE_RESOURCE_GROUP=
declare -x AZURE_KEYVAULT_ENDPOINT=
declare -x AZURE_STORAGE_ACCOUNT_NAME=
declare -x CCRP_DID=

# For key import:
declare -x CONTRACT_SERVICE_URL=
declare -x TOOLS_HOME=$REPO_ROOT/external/confidential-sidecar-containers/tools

# Export all variables to make them available to other scripts
export SCENARIO
export REPO_ROOT
export CONTAINER_REGISTRY
export AZURE_LOCATION
export AZURE_SUBSCRIPTION_ID
export AZURE_RESOURCE_GROUP
export AZURE_KEYVAULT_ENDPOINT
export AZURE_STORAGE_ACCOUNT_NAME
export CCRP_DID
export CONTRACT_SERVICE_URL
export TOOLS_HOME
