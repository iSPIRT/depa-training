variable "resource_group_location" {
  default     = "eastus"
  description = "Location of the resource group"
}

variable "azure_keyvault_endpoint" {
  type    = string
  default = "vault.azure.net"
}

variable "depa_keyvault_endpoint" {
  type    = string
  default = "depakv.vault.azure.net"
}

variable "contract_service_url" {
  description = "URL of the contract service"
  default     = "https://20.42.118.35:8000/parameters" # Provide a default if not set
}
  #default     = "https://contract-service.westeurope.cloudapp.azure.com:8000/parameters" # Provide a default if not set


variable "depa_home" {
  description = "Path to the directory containing the 'securitypolicydigest' tool"
  default     = "/Users/mukeshjoshi/gitprojects/depa-training" # No default; make it required if needed
}

variable "container_registry" {
  description = "Path to the directory containing the containers"
  default     = "ispirt" # No default; make it required if needed
}

# variable "tools_home" {
#   description = "Path to the directory containing the 'securitypolicydigest' tool"
#   default     = var.depa_home"/external/confidential-sidecar-containers/tools"  # No default; make it required if needed
# }

# variable "policy_home" {
#   description = "Path to the directory containing the template policy"
#   default     = var.depa_home"/scenarios/covid/policy"  # No default; make it required if needed
# }

# variable "policy_template" {
#   description = "File containing the policy template json"
#   default     = var.policy_home"policy-in-template.json"  # No default; make it required if needed
# }
