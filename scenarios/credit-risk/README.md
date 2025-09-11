# Home Credit Default Risk Prediction

## Scenario Type

| Scenario name | Scenario type | Task type | Privacy | No. of TDPs* | Data type (format) | Model type (format) | Join type (No. of datasets) |
|--------------|---------------|-----------------|--------------|-----------|------------|------------|------------|
| Credit Risk | Training - Classical ML | Binary Classification | Differentially Private | 4 | PII tabular data (Parquet) | XGBoost (JSON) | Horizontal (6)|

---

## Scenario Description

This scenario involves training an **XGBoost** model on the [Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk) datasets [[1, 2]](README.md#references). We frame this scenario as involving four Training Data Providers (TDPs) - **Bank A** providing data for clients' (i) credit applications, (ii) previous applications and (iii) payment installments, **Bank B** providing data on (i) credit card balance, the **Credit Bureau** providing data on (i) previous loans, and a **Fintech** providing data on (i) point of sale (POS) cash balance. Here, **Bank A** is also the Training Data Consumer (TDC) who wishes to train the model on the joined datasets, in order to build a default risk prediction model.

The end-to-end training pipeline consists of the following phases:

1. Data pre-processing
2. Data packaging, encryption and upload
3. Encryption key import with key release policies
4. Deployment and execution of CCR
5. Trained model decryption

## Build container images

Build container images required for this sample as follows.

```bash
export SCENARIO=credit-risk
export REPO_ROOT="$(git rev-parse --show-toplevel)"
cd $REPO_ROOT/scenarios/$SCENARIO
./ci/build.sh
```

This script builds the following container images.

- ```preprocess-bank-a```: Container for pre-processing Bank A's dataset.
- ```preprocess-bank-b```: Container for pre-processing Bank B's dataset.
- ```preprocess-bureau```: Container for pre-processing Bureau's dataset.
- ```preprocess-fintech```: Container for pre-processing Fintech's dataset.

Alternatively, you can pull and use pre-built container images from the iSPIRT container registry by setting the following environment variable. Docker hub has started throttling which may affect the upload/download time, especially when images are bigger size. So, It is advisable to use other container registries. We are using Azure container registry (ACR) as shown below:

```bash
export CONTAINER_REGISTRY=ispirt.azurecr.io
cd $REPO_ROOT/scenarios/$SCENARIO
./ci/pull-containers.sh
```

## Data pre-processing

The folder ```scenarios/credit-risk/src``` contains scripts for downloading and pre-processing the datasets. Acting as a Training Data Provider (TDP), prepare your datasets.

Since these datasets are downloaded from [Kaggle.com](https://kaggle.com), set your Kaggle credentials as environment variables before running the preprocess scripts. The Kaggle credentials can be obtained from your Kaggle account settings > API > Create new token.

```bash
export KAGGLE_USERNAME=<kaggle-username>
export KAGGLE_KEY=<kaggle-key>
```

Then, run the preprocess script.

```bash
cd $REPO_ROOT/scenarios/$SCENARIO/deployment/local
./preprocess.sh
```

The datasets are saved to the [data](./data/) directory.

## Deploy locally

Assuming you have cleartext access to all the datasets, you can train the model _locally_ as follows:

```bash
./train.sh
```

The script joins the datasets and trains the model using a pipeline configuration. To modify the various components of the training pipeline, you can edit the training config files in the [config](./config/) directory. The training config files are used to create the pipeline configuration ([pipeline_config.json](./config/pipeline_config.json)) created by consolidating all the TDC's training config files, namely the [model config](./config/model_config.json), [dataset config](./config/dataset_config.json), [loss function config](./config/loss_config.json), [training config](./config/train_config_template.json), [evaluation config](./config/eval_config.json), and if multiple datasets are used, the [data join config](./config/join_config.json). These enable the TDC to design highly customized training pipelines without requiring review and approval of new custom code for each use case — reducing risks from potentially malicious or non-compliant code. The consolidated pipeline configuration is then attested against the signed contract using the TDP's policy-as-code. If approved, it is executed in the CCR to train the model, which we will deploy in the next section.

```mermaid
flowchart TD

    subgraph Config Files
        C1[model_config.json]
        C2[dataset_config.json]
        C3[loss_config.json]
        C4[train_config_template.json]
        C5[eval_config.json]
        C6[join_config.json]
    end

    B[Consolidated into <br/> pipeline_config.json]

    C1 --> B
    C2 --> B
    C3 --> B
    C4 --> B
    C5 --> B
    C6 --> B

    B --> D[Attested against contract<br/>using policy-as-code]
    D --> E{Approved?}
    E -- Yes --> F[CCR training begins]
    E -- No --> H[Rejected: fix config]
```

If all goes well, you should see output similar to the following output, and the trained model and evaluation metrics will be saved under the folder [output](./modeller/output).

```
train-1  | Joined datasets: ['credit_applications', 'previous_applications', 'payment_installments', 'bureau_records', 'pos_cash_balance', 'credit_card_balance']
train-1  | Loaded dataset splits | train: (6038, 63) | val: (755, 63) | test: (755, 63)
train-1  | Trained Gradient Boosting model with 250 boosting rounds | Epsilon: 4.0
train-1  | Saved model to /mnt/remote/output
train-1  | Evaluation Metrics: {'accuracy': 0.5231788079470199, 'roc_auc': 0.47444826338639656}
train-1  | Non-DP Evaluation Metrics: {'accuracy': 0.9152317880794701, 'roc_auc': 0.6496472503617945}
train-1  | CCR Training complete!
train-1  |
train-1 exited with code 0
```

## Deploy on CCR

In a more realistic scenario, these datasets will not be available in the clear to the TDC, and the TDC will be required to use a CCR for training her model. The following steps describe the process of sharing encrypted datasets with TDCs and setting up a CCR in Azure for training. Please stay tuned for CCR on other cloud platforms.

To deploy in Azure, you will need the following.

- Docker Hub account to store container images. Alternatively, you can use pre-built images from the ```iSPIRT``` container registry.
- [Azure Key Vault](https://azure.microsoft.com/en-us/products/key-vault/) to store encryption keys and implement secure key release to CCR. You can either use Azure Key Vault Premium (lower cost), or [Azure Key Vault managed HSM](https://learn.microsoft.com/en-us/azure/key-vault/managed-hsm/overview) for enhanced security. Please see instructions below on how to create and set up your AKV instance.
- Valid Azure subscription with sufficient access to create key vault, storage accounts, storage containers, and Azure Container Instances (ACI).

If you are using your own development environment instead of a dev container or codespaces, you will need to install the following dependencies.

- [Azure CLI](https://learn.microsoft.com/en-us/cli/azure/install-azure-cli-linux).  
- [Azure CLI Confidential containers extension](https://learn.microsoft.com/en-us/cli/azure/confcom?view=azure-cli-latest). After installing Azure CLI, you can install this extension using ```az extension add --name confcom -y```
- [Go](https://go.dev/doc/install). Follow the instructions to install Go. After installing, ensure that the PATH environment variable is set to include ```go``` runtime.
- ```jq```. You can install jq using ```sudo apt-get install -y jq```

We will be creating the following resources as part of the deployment.

- Azure Key Vault
- Azure Storage account
- Storage containers to host encrypted datasets
- Azure Container Instances (ACI) to deploy the CCR and train the model

### 1. Push Container Images

Pre-built container images are available in iSPIRT's container registry, which can be pulled by setting the following environment variable.

```bash
export CONTAINER_REGISTRY=ispirt.azurecr.io
```

If you wish to use your own container images, login to docker hub (or your container registry of choice) and then build and push the container images to it, so that they can be pulled by the CCR. This is a one-time operation, and you can skip this step if you have already pushed the images to your container registry.

```bash
export CONTAINER_REGISTRY=<container-registry-name>
docker login -u <docker-hub-username> -p <docker-hub-password> ${CONTAINER_REGISTRY}
cd $REPO_ROOT
./ci/push-containers.sh
cd $REPO_ROOT/scenarios/$SCENARIO
./ci/push-containers.sh
```

> **Note:** Replace `<container-registry-name>`, `<docker-hub-username>` and `<docker-hub-password>` with your container registry name, docker hub username and password respectively. Preferably use registry services other than Docker Hub as throttling restrictions will cause delays (or) image push/pull failures.

### 2. Create Resources

First, set up the necessary environment variables for your deployment.

```bash
az login

export SCENARIO=credit-risk
export CONTAINER_REGISTRY=ispirt.azurecr.io
export AZURE_LOCATION=northeurope
export AZURE_SUBSCRIPTION_ID=<azure-subscription-id>
export AZURE_RESOURCE_GROUP=<resource-group-name>
export AZURE_KEYVAULT_ENDPOINT=<key-vault-name>.vault.azure.net
export AZURE_STORAGE_ACCOUNT_NAME=<storage-account-name>

export AZURE_BANK_A_CONTAINER_NAME=bankacontainer
export AZURE_BANK_B_CONTAINER_NAME=bankbcontainer
export AZURE_BUREAU_CONTAINER_NAME=bureaucontainer
export AZURE_FINTECH_CONTAINER_NAME=fintechcontainer
export AZURE_OUTPUT_CONTAINER_NAME=outputcontainer
```

Alternatively, you can edit the values in the [export-variables.sh](./export-variables.sh) script and run it to set the environment variables.

```bash
./export-variables.sh
source export-variables.sh
```

Azure Naming Rules:
- Resource Group:
  - 1–90 characters
  - Letters, numbers, underscores, parentheses, hyphens, periods allowed
  - Cannot end with a period (.)
  - Case-insensitive, unique within subscription\
- Key Vault:
  - 3-24 characters
  - Globally unique name
  - Lowercase letters, numbers, hyphens only
  - Must start and end with letter or number
- Storage Account:
  - 3-24 characters
  - Globally unique name
  - Lowercase letters and numbers only
- Storage Container:
  - 3-63 characters
  - Lowercase letters, numbers, hyphens only
  - Must start and end with a letter or number
  - No consecutive hyphens
  - Unique within storage account

---

**Important:**

The values for the environment variables listed below must precisely match the namesake environment variables used during contract signing (next step). Any mismatch will lead to execution failure.

-  `SCENARIO`
- `AZURE_KEYVAULT_ENDPOINT`
- `CONTRACT_SERVICE_URL`
- `AZURE_STORAGE_ACCOUNT_NAME`
- `AZURE_BANK_A_CONTAINER_NAME`
- `AZURE_BANK_B_CONTAINER_NAME`
- `AZURE_BUREAU_CONTAINER_NAME`
- `AZURE_FINTECH_CONTAINER_NAME`

---
With the environment variables set, we are ready to create the resources -- Azure Key Vault and Azure Storage containers.

```bash
cd $REPO_ROOT/scenarios/$SCENARIO/deployment/azure
./1-create-storage-containers.sh
./2-create-akv.sh
```
---

### 3\. Contract Signing

Navigate to the [contract-ledger](https://github.com/kapilvgit/contract-ledger/blob/main/README.md) repository and follow the instructions for contract signing.

Once the contract is signed, export the contract sequence number as an environment variable in the same terminal where you set the environment variables for the deployment.

```bash
export CONTRACT_SEQ_NO=<contract-sequence-number>
```

---

### 4\. Data Encryption and Upload

Using their respective keys, the TDPs and TDC encrypt their datasets and models (respectively) and upload them to the Storage containers created in the previous step.

Navigate to the [Azure deployment](./deployment/azure/) directory and execute the scripts for key import, data encryption and upload to Azure Blob Storage, in preparation of the CCR deployment.

The import-keys script generates and imports encryption keys into Azure Key Vault with a policy based on [policy-in-template.json](./policy/policy-in-template.json). The policy requires that the CCRs run specific containers with a specific configuration which includes the public identity of the contract service. Only CCRs that satisfy this policy will be granted access to the encryption keys. The generated keys are available as files with the extension `.bin`.

```bash
export CONTRACT_SERVICE_URL=https://depa-training-contract-service.centralindia.cloudapp.azure.com:8000
export TOOLS_HOME=$REPO_ROOT/external/confidential-sidecar-containers/tools

./3-import-keys.sh
```

The data and model are then packaged as encrypted filesystems by the TDPs and TDC using their respective keys, which are saved as `.img` files.

```bash
./4-encrypt-data.sh
```

The encrypted data and model are then uploaded to the Storage containers created in the previous step. The `.img` files are uploaded to the Storage containers as blobs.

```bash
./5-upload-encrypted-data.sh
```

---

### 5\. CCR Deployment

With the resources ready, we are ready to deploy the Confidential Clean Room (CCR) for executing the privacy-preserving model training.

```bash
export CONTRACT_SEQ_NO=<contract-sequence-number>
./deploy.sh -c $CONTRACT_SEQ_NO -p ../../config/pipeline_config.json
```

Set the `$CONTRACT_SEQ_NO` variable to the exact value of the contract sequence number (of format 2.XX). For example, if the number was 2.15, export as:

```bash
export CONTRACT_SEQ_NO=15
```

This script will deploy the container images from your container registry, including the encrypted filesystem sidecar. The sidecar will generate an SEV-SNP attestation report, generate an attestation token using the Microsoft Azure Attestation (MAA) service, retrieve dataset, model and output encryption keys from the TDP and TDC's Azure Key Vault, train the model, and save the resulting model into TDC's output filesystem image, which the TDC can later decrypt.

**Note:** The completion of this script's execution simply creates a CCR instance, and doesn't indicate whether training has completed or not. The training process might still be ongoing. Poll the container logs (see below) to track progress until training is complete.

### 6\. Monitor Container Logs

Use the following commands to monitor the logs of the deployed containers. You might have to repeatedly poll this command to monitor the training progress:

```bash
az container logs \
  --name "depa-training-$SCENARIO" \
  --resource-group "$AZURE_RESOURCE_GROUP" \
  --container-name depa-training
```

You will know training has completed when the logs print "CCR Training complete!".

#### Troubleshooting

In case training fails, you might want to monitor the logs of the encrypted storage sidecar container to see if the encryption process completed successfully:

```bash
az container logs --name depa-training-$SCENARIO --resource-group $AZURE_RESOURCE_GROUP --container-name encrypted-storage-sidecar
```

And to further debug, inspect the logs of the encrypted filesystem sidecar container:

```bash
az container exec \
  --resource-group $AZURE_RESOURCE_GROUP \
  --name depa-training-$SCENARIO \
  --container-name encrypted-storage-sidecar \
  --exec-command "/bin/sh"
```

Once inside the sidecar container shell, view the logs:

```bash
cat log.txt
```
Or inspect the individual mounted directories in `mnt/remote/`:

```bash
cd mnt/remote && ls
```

### 6\. Download and Decrypt Model

Once training has completed successfully (The training container logs will mention it explicitly), download and decrypt the trained model and other training outputs.

```bash
./6-download-decrypt-model.sh
```

The outputs will be saved to the [output](./modeller/output/) directory.

To check if the trained model is fresh, you can run the following command:

```bash
stat $REPO_ROOT/scenarios/$SCENARIO/modeller/output/trained_model.json
```

---
### Clean-up

You can use the following command to delete the resource group and clean-up all resources used in the demo. Alternatively, you can navigate to the Azure portal and delete the resource group created for this demo.

```bash
az group delete --yes --name $AZURE_RESOURCE_GROUP
```

## References

[1] [Anna Montoya, inversion, KirillOdintsov, and Martin Kotek. Home Credit Default Risk. https://kaggle.com/competitions/home-credit-default-risk, 2018. Kaggle.](https://www.kaggle.com/c/home-credit-default-risk)

[2] [XGBoost](https://xgboost.readthedocs.io/en/stable/)