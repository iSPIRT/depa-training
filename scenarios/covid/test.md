# COVID predictive modelling 

This hypothetical scenario demonstrates how a deep learning model can be trained for COVID predictive analytics using the join of multiple PII-sensitive datasets. The Training Data Consumer (TDC) building the model enters into a contractual agreement with multiple Training Data Providers (TDPs) — here, hypothetically ICMR, COWIN, and a State War Room — each contributing datasets. The model is trained on the joined datasets in a data-blind manner within a Confidential Clean Room (CCR), with privacy guarantees using Differential Privacy.

For demonstration purposes, this scenario uses synthetically generated mock datasets and a simple neural network model architecture.

_The datasets and model are for illustrative purposes only; none of these organizations have been involved in contributing to the code or datasets._

The end-to-end training pipeline consists of the following phases:

1. Data pre-processing and de-identification.
2. Packaging, encryption and upload of data and model
3. Model packaging, encryption and upload
4. Encryption key import with key release policies
5. Deployment and execution of CCR
6. Model decryption

## Build container images

Build container images required for this sample as follows:

```bash
cd ~/depa-training/scenarios/covid
./ci/build.sh
```

This script builds the following container images:

- `preprocess-icmr, preprocess-cowin, preprocess-index`: Containers that pre-process and de-identify the datasets.
- `covid-model-save`: Container that saves the base model to be trained.

Alternatively, you can pull and use pre-built container images from the ispirt container registry by setting the following environment variable. Docker hub has started throttling which may effect the upload/download time, especially when images are bigger size. So, It is advisable to use other container registries. We are using Azure container registry (ACR) as shown below

```bash
export CONTAINER_REGISTRY=depatraindevacr.azurecr.io
cd ~/depa-training/scenarios/covid
./ci/pull-containers.sh
```

## Data pre-processing and de-identification

The folders ```scenarios/covid/data``` contains three sample training datasets. Acting as TDPs for these datasets, run the following scripts to de-identify the datasets. 

```bash
cd ~/depa-training/scenarios/covid/deployment/docker
./preprocess.sh
```

The datasets are saved to the [data](./data/) directory.

## Prepare model for training

Next, acting as a Training Data Consumer (TDC), define and save your base model for training using the following script:

```bash
./save-model.sh
```

This script will save the base model to the [models](./modeller/models) directory.

## Deploy locally

Assuming you have cleartext access to all the datasets, you can train the model _locally_ as follows:

```bash
./train.sh
```

The script joins the datasets and trains the model using a pipeline configuration defined in [pipeline_config.json](./config/pipeline_config.json). The training process uses:

- Simple feed-forward neural network model architecture for binary classification.
- Differential Privacy to prevent reconstruction & membership inference attacks, using the Opacus library.
- PyTorch 

If all goes well, you should see training progress output similar to:

```
Epoch [1/5], Loss: 0.4849
Epoch [2/5], Loss: 0.0821
Epoch [3/5], Loss: 0.0489
Epoch [4/5], Loss: 0.0138
Epoch [5/5], Loss: 0.0080
```

The trained model will be saved under the [output](./modeller/output/) directory.

Now that you have tested the training locally, let's move on the actual execution using a Confidential Clean Room (CCR) equipped with confidential computing, key release policies, and contract-based access control.

## Deploy on CCR

Follow these steps to deploy and run the DEPA-Training solution on Azure's Container Instances (ACI) with confidential computing.

### 1. Azure and Docker Login

```bash
az login
docker login
```

---

### 2. Export Environment Variables

Set up the necessary environment variables for your deployment.

```bash
export SCENARIO=covid
export CONTAINER_REGISTRY=depatraindevacr.azurecr.io
export AZURE_LOCATION=northeurope
export AZURE_SUBSCRIPTION_ID=<azure-subscription-id>
export AZURE_RESOURCE_GROUP=<resource-group-name>
export AZURE_KEYVAULT_ENDPOINT=<key-vault-endpoint-name>.vault.azure.net
export AZURE_STORAGE_ACCOUNT_NAME=<storage-account-name>

export AZURE_ICMR_CONTAINER_NAME=icmrcontainer
export AZURE_COWIN_CONTAINER_NAME=cowincontainer
export AZURE_INDEX_CONTAINER_NAME=indexcontainer
export AZURE_MODEL_CONTAINER_NAME=modelcontainer
export AZURE_OUTPUT_CONTAINER_NAME=outputcontainer

export CONTRACT_SERVICE_URL=https://depa-contract-service.southindia.cloudapp.azure.com:8000
export TOOLS_HOME=~/depa-training/external/confidential-sidecar-containers/tools
```

---

**Important:**

The values for the below environment variables must precisely match the namesake environment variables used during contract signing. Any mismatch will lead to execution failure.
-  `SCENARIO`
- `AZURE_KEYVAULT_ENDPOINT`
- `CONTRACT_SERVICE_URL`
- `AZURE_STORAGE_ACCOUNT_NAME`
- `AZURE_ICMR_CONTAINER_NAME`
- `AZURE_COWIN_CONTAINER_NAME`
- `AZURE_INDEX_CONTAINER_NAME`

---

### 3\. Data Preparation and Key Management

Navigate to the [ACI deployment](./deployment/aci/) directory and execute the scripts for storage container creation, Azure Key Vault setup, key import, data encryption and upload to Azure Blob Storage, in preparation of the CCR deployment.

```bash
cd ~/depa-training/scenarios/$SCENARIO/deployment/aci

./1-create-storage-containers.sh

./2-create-akv.sh

./3-import-keys.sh

./4-encrypt-data.sh

./5-upload-encrypted-data.sh
```

### 4\. ACI Deployment

With the resources ready, we are ready to deploy the Confidential Clean Room (CCR) for executing the privacy-preserving model training.

```bash
export CONTRACT_SEQ_NO=$(./get-contract-seq-no.sh)

./deploy.sh -c $CONTRACT_SEQ_NO -p ../../config/pipeline_config.json

```

**Note:** if the contract-ledger repository is also located at the root of the same environment where this depa-training repo is, the `$CONTRACT_SEQ_NO` variable automatically picks up the sequence number of the latest contract that was signed between the TDPs and TDC.

If not, manually set the `$CONTRACT_SEQ_NO` variable to the exact value of the contract sequence number (of format 2.XX). For example, if the number was 2.15, export as:

```bash
export CONTRACT_SEQ_NO=15
```

**Note:** The completion of this script's execution simply creates a CCR instance, and doesn't indicate whether training has completed or not. The training process might still be ongoing. Monitor the container logs to track progress until training is complete.

### 5\. Monitor Container Logs

Use the following commands to monitor the logs of the deployed containers. You might have to repeatedly poll this command to monitor the training progress:

```bash
az container logs \
  --name "depa-training-$SCENARIO" \
  --resource-group "$AZURE_RESOURCE_GROUP" \
  --container-name depa-training
```

#### Troubleshooting

In case training fails, you might want to monitor the logs of the encrypted storage sidecar container to see if the encryption process completed successfully:

```bash
az container logs --name depa-training-$SCENARIO --resource-group $AZURE_RESOURCE_GROUP --container-name encrypted-storage-sidecar
```

And to further debug, inspect the logs of the encrypted filesystem sidecar container:

```
az container exec \
  --resource-group $AZURE_RESOURCE_GROUP \
  --name depa-training-$SCENARIO \
  --container-name encrypted-storage-sidecar \
  --exec-command "/bin/sh"
```

Once inside the sidecar container shell, view the logs:

```
cat log.txt
```
Or inspect the individual mounted directories in `mnt/remote/`:

```
cd mnt/remote && ls
```

### 6\. Download and Decrypt Model

Once training has completed succesfully (The training container logs will mention it explicitly), download and decrypt the trained model and other training outputs.

```bash
./6-download-decrypt-model.sh
```

The outputs will be saved to the [output](./modeller/output/) directory.

To check if the trained model is fresh, you can run the following command:

```bash
stat ~/depa-training/scenarios/$SCENARIO/modeller/output/trained_model.pth
```

---
### Clean-up

You can use the following command to delete the resource group and clean-up all resources used in the demo. Alternatively, you can navigate to the Azure portal and delete the resource group created for this demo.

```
az group delete --yes --name $AZURE_RESOURCE_GROUP
```