# CI Workflows

This repository contains several CI workflows designed for deployment in a self-hosted runner. Below is a description of each workflow and its purpose.

## Workflows

### `ci-build.yml` 
**Trigger:** Automatically on every pull request or Manually

- Builds the `depa-training-encfs`, `depa-training`, and other containers required for the COVID scenario.
- Runs containers locally to perform preprocessing, save the model, and train the model.

### `ci.yml`
**Trigger:** Manually

- Prepares data and model for deployment.
- Creates Azure Storage and Key Vault if they do not already exist.
- Imports data and model encryption keys with key release policies.
- Encrypts the data and model.
- Uploads the encrypted data and model to Azure Storage.
- Deploys CCR on Azure Container Instances (ACI) and trains the model.

### `ci-local.yml`
**Trigger:** Manually

- Pulls containers from Azure Container Registry (ACR).
- Runs containers locally to perform preprocessing, save the model, and train the model.

### `release.yml`
**Trigger:** Release Event 

- Builds the `depa-training-encfs`, `depa-training`, contract service container, and other containers required for the COVID scenario.
- Pushes the built containers to ACR.

### `contract-service.yml`
**Trigger:** Manually

- Builds the contract service container.
- Deploys the contract service.

## Steps to Deploy Self-Hosted Runner

1. **Set up Recommended OIDC Authentication:**
   - Follow the [official guide](https://learn.microsoft.com/en-us/azure/app-service/deploy-github-actions?tabs=openid%2Cpython%2Caspnetcore#set-up-a-github-actions-workflow-manually) to authenticate GitHub Actions with Azure services using the OIDC approach.

2. **Assign Necessary Permissions:**
   - For the service principal created during the above step (or an external one), assign the following permissions:
     - Contributor
     - Custom role with `Microsoft.Authorization/GetRoleAssignment` and `Microsoft.Authorization/CreateRoleAssignment`

3. **Create a New Self-Hosted Runner:**
   - Navigate to `Settings` > `Actions` > `Runners` in your GitHub repository.
   - Create a "New self-hosted Runner".

4. **Set Up the Azure VM:**
   - Create an Azure VM with at least 16GB RAM, 4 CPUs, and 128GB SSD.
   - Follow the instructions provided in the self-hosted runner setup to configure the action runner on your Azure VM.

---

By following these instructions, you can set up and utilize the CI workflows in your self-hosted runner to automate and manage the deployment processes for your projects.

