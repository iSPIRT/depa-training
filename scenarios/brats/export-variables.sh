#!/bin/bash

# Azure Naming Rules:
#
# Resource Group:
# - 1–90 characters
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

# For cloud resource creation:
declare -x SCENARIO=brats
declare -x REPO_ROOT="$(git rev-parse --show-toplevel)"
declare -x CONTAINER_REGISTRY=ispirt.azurecr.io
declare -x AZURE_LOCATION=<azure-location>
declare -x AZURE_SUBSCRIPTION_ID=
declare -x AZURE_RESOURCE_GROUP=
declare -x AZURE_KEYVAULT_ENDPOINT=
declare -x AZURE_STORAGE_ACCOUNT_NAME=

declare -x AZURE_BRATS_A_CONTAINER_NAME=bratsacontainer
declare -x AZURE_BRATS_B_CONTAINER_NAME=bratsbcontainer
declare -x AZURE_BRATS_C_CONTAINER_NAME=bratsccontainer
declare -x AZURE_BRATS_D_CONTAINER_NAME=bratsdcontainer
declare -x AZURE_MODEL_CONTAINER_NAME=modelcontainer
declare -x AZURE_OUTPUT_CONTAINER_NAME=outputcontainer

# For key import:
declare -x CONTRACT_SERVICE_URL=https://<contract-service-url>:8000
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
export AZURE_BRATS_A_CONTAINER_NAME
export AZURE_BRATS_B_CONTAINER_NAME
export AZURE_BRATS_C_CONTAINER_NAME
export AZURE_BRATS_D_CONTAINER_NAME
export AZURE_MODEL_CONTAINER_NAME
export AZURE_OUTPUT_CONTAINER_NAME
export CONTRACT_SERVICE_URL
export TOOLS_HOME