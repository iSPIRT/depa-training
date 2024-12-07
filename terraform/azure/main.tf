
# Set variables for your configuration (environment variables in this case)
variable "AZURE_RESOURCE_GROUP" {}
variable "AZURE_KEYVAULT_NAME" {}
variable "AZURE_STORAGE_ACCOUNT_NAME" {}
variable "AZURE_ICMR_CONTAINER_NAME" {}
variable "AZURE_COWIN_CONTAINER_NAME" {}
variable "AZURE_INDEX_CONTAINER_NAME" {}
variable "AZURE_MODEL_CONTAINER_NAME" {}
variable "AZURE_OUTPUT_CONTAINER_NAME" {}

locals {
  azure_resource_group           = var.AZURE_RESOURCE_GROUP
  cce_policy_hash                = "YOUR_CCE_POLICY_HASH"                # Get this from your env
  keyid                          = "YOUR_KEYID"                          # Get this from your env
  azure_akv_key_type             = "YOUR_AZURE_AKV_KEY_TYPE"             # Get this from your env
  azure_akv_key_derivation_label = "YOUR_AZURE_AKV_KEY_DERIVATION_LABEL" # Get this from your env
  policy_template                = "./policy-in-template-tf.json"
  key_type                       = can(regex(".vault.azure.net", var.azure_keyvault_endpoint)) ? "RSA-HSM" : "oct-HSM"
}
# Create Resource Group
resource "azurerm_resource_group" "covid_data_rg" {
  name     = var.AZURE_RESOURCE_GROUP
  location = var.resource_group_location
}

# Data Source: Fetch Current Azure Account Information
data "azurerm_client_config" "current" {}

# Create Storage Account
resource "azurerm_storage_account" "covid_data_sa" {
  name                     = var.AZURE_STORAGE_ACCOUNT_NAME
  resource_group_name      = azurerm_resource_group.covid_data_rg.name
  location                 = azurerm_resource_group.covid_data_rg.location
  account_tier             = "Standard"
  account_replication_type = "LRS"
}

# Create Storage Containers
resource "azurerm_storage_container" "icmr_container" {
  name                  = var.AZURE_ICMR_CONTAINER_NAME
  storage_account_name  = azurerm_storage_account.covid_data_sa.name
  container_access_type = "private"
}

resource "azurerm_storage_container" "cowin_container" {
  name                  = var.AZURE_COWIN_CONTAINER_NAME
  storage_account_name  = azurerm_storage_account.covid_data_sa.name
  container_access_type = "private"
}

resource "azurerm_storage_container" "index_container" {
  name                  = var.AZURE_INDEX_CONTAINER_NAME
  storage_account_name  = azurerm_storage_account.covid_data_sa.name
  container_access_type = "private"
}

resource "azurerm_storage_container" "model_container" {
  name                  = var.AZURE_MODEL_CONTAINER_NAME
  storage_account_name  = azurerm_storage_account.covid_data_sa.name
  container_access_type = "private"
}

resource "azurerm_storage_container" "output_container" {
  name                  = var.AZURE_OUTPUT_CONTAINER_NAME
  storage_account_name  = azurerm_storage_account.covid_data_sa.name
  container_access_type = "private"
}



# Create Azure Key Vault
resource "azurerm_key_vault" "covid_keyvault" {
  name                      = var.AZURE_KEYVAULT_NAME
  resource_group_name       = azurerm_resource_group.covid_data_rg.name # Assuming your resource group is already created
  location                  = azurerm_resource_group.covid_data_rg.location
  sku_name                  = "premium"
  tenant_id                 = data.azurerm_client_config.current.tenant_id
  enable_rbac_authorization = true
}

# Create Role Assignments
resource "azurerm_role_assignment" "keyvault_crypto_officer" {
  scope                = azurerm_key_vault.covid_keyvault.id
  role_definition_name = "Key Vault Crypto Officer"
  principal_id         = data.azurerm_client_config.current.object_id
}

resource "azurerm_role_assignment" "keyvault_crypto_user" {
  scope                = azurerm_key_vault.covid_keyvault.id
  role_definition_name = "Key Vault Crypto User"
  principal_id         = data.azurerm_client_config.current.object_id
}

# Resource Group
resource "azurerm_resource_group" "confidential_containers_rg" {
  name     = "confidential-containers-rg"
  location = "East US" # Replace with your desired Azure region
}




# # Container Registry (Optional)
# resource "azurerm_container_registry" "confidential_registry" {
#   name                = "confidentialregistry"
#   resource_group_name = azurerm_resource_group.confidential_containers_rg.name
#   location            = azurerm_resource_group.confidential_containers_rg.location
#   sku                 = "Standard"
# }

# # Container Group
# resource "azurerm_container_group" "confidential_container" {
#   name                = "confidentialcontainer"
#   resource_group_name = azurerm_resource_group.confidential_containers_rg.name
#   location            = azurerm_resource_group.confidential_containers_rg.location
#   sku                 = "Confidential"
#   os_type             = "Linux"  # Or "Windows"

#   container {
#     name   = "mycontainer"
#     image  = "nginx:latest"   # Replace with your container image
#     cpu    = 1
#     memory = 1
#     ports {
#       port     = 8080        # Internal port within the container
#       protocol = "TCP"    
#     }

#   }

#   container {
#     name   = "mycontainer2"
#     image  = "nginx:latest"   # Replace with your container image
#     cpu    = 1
#     memory = 1
#     ports {
#       port     = 8081       # Internal port within the container
#       protocol = "TCP"    
#     }

#   }
# }



data "external" "azure_keyvault_token" {
  program = [
    "bash",
    "-c",
    "az account get-access-token --resource https://vault.azure.net | jq '{accessToken: .accessToken}'"
  ]
}

output "azure_access_token" {
  value     = data.external.azure_keyvault_token.result.accessToken
  sensitive = true
}


data "http" "contract_service_parameters" {
  url      = var.contract_service_url # Assuming you have the URL in a variable
  insecure = true                     # Disable SSL verification (use with caution)
}

locals {
  contract_service_parameters_encoded = base64encode(data.http.contract_service_parameters.response_body)
}

resource "local_file" "policy_file" {
  content = templatefile(local.policy_template, {
    CONTAINER_REGISTRY          = var.container_registry,
    CONTRACT_SERVICE_PARAMETERS = local.contract_service_parameters_encoded
  })
  filename = "${path.module}/policy.json"

}


#########=====

 data "external" "get_cce_policy" {
   program = [
     "bash",
     "-c",
     "az confcom acipolicygen -i ${local_file.policy_file.filename} --debug"
   ]
 }

# output "cce_policy_result" {
#  value     = data.external.get_cce_policy.result
#  sensitive = false
# }



# # Null resource to trigger command execution
# resource "null_resource" "generate_cce_policy" {
#   triggers = {
#     policy_content = local_file.policy_file.content # Trigger on policy changes
#   }

#   # Local exec provisioner to run the Azure CLI command
#   provisioner "local-exec" {
#     command = "az confcom acipolicygen -i ${local_file.policy_file.filename} --debug-mode > cce_policy.txt"
#   }
# }

# resource "local_file" "cce_policy_file" {
#   content  = null_resource.generate_cce_policy.triggers.CCE_POLICY
#   filename = "cce_policy.txt"

#   depends_on = [
#     null_resource.generate_cce_policy,
#   ]
# }

# # (Optional) Output the generated policy
# output "cce_policy" {
#   value     = local.cce_policy_file
#   sensitive = true # Mark as sensitive if the policy contains confidential data
# }


# # Local variable to store the generated policy
# locals {
#   cce_policy_file = null_resource.generate_cce_policy.triggers.policy_content
# }


#########=====
output "env_details" {
  value = {
    project_home                        = var.depa_home
    azure_resource_group                = local.azure_resource_group
    azure_keyvault_endpoint             = var.azure_keyvault_endpoint
    key_type                            = local.key_type
    policy_template                     = local.policy_template
    contract_service_parameters_encoded = local.contract_service_parameters_encoded
    # cce_policy                               = local.cce_policy_result, sensitive = false # Mark as sensitive if the policy contains confidential data
  }
  description = "Details of the env"
}







