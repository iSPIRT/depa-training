# Convolution Neural Network using MNIST

This scenario involves training a CNN using the MNIST dataset. It involves one training data provider (TDP), and a TDC who wishes the train a model. 

The end-to-end training pipeline consists of the following phases. 

1. Data pre-processing and de-identification
2. Data packaging, encryption and upload
3. Model packaging, encryption and upload 
4. Encryption key import with key release policies
5. Deployment and execution of CCR
6. Model decryption 

## Build container images

Build container images required for this sample as follows. 

```bash
cd scenarios/mnist
./ci/build.sh
```

These scripts build the following containers. 

- ```depa-mnist-preprocess```: Container for pre-processing MNIST dataset. 
- ```depa-mnist-save-model```: Container that saves the model to be trained in ONNX format. 

## Data pre-processing and de-identification

The folders ```scenarios/mnist/data``` contains scripts for downloading and pre-processing the MNIST dataset. Acting as a TDP for this dataset, run the following script. 

```bash
cd scenarios/mnist/deployment/docker
./preprocess.sh
```

## Prepare model for training

Next, acting as a TDC, save a sample model using the following script. 

```bash
./save-model.sh
```

This script will save the model as ```scenarios/mnist/data/model/model.onnx.```

## Deploy locally

Assuming you have cleartext access to the pre-processed dataset, you can train a CNN as follows. 

```bash
./train.sh
```
The script trains a model using a pipeline configuration defined in [pipeline_config.json](./config/pipeline_config.json). If all goes well, you should see output similar to the following output, and the trained model will be saved under the folder `/tmp/output`. 

```
docker-train-1  | /usr/local/lib/python3.9/dist-packages/torchvision/io/image.py:13: UserWarning: Failed to load image Python extension: 'libc10_cuda.so: cannot open shared object file: No such file or directory'If you don't plan on using image functionality from `torchvision.io`, you can ignore this warning. Otherwise, there might be something wrong with your environment. Did you have `libjpeg` or `libpng` installed before building `torchvision` from source?
docker-train-1  |   warn(
docker-train-1  | /usr/local/lib/python3.9/dist-packages/onnx2pytorch/convert/layer.py:30: UserWarning: The given NumPy array is not writable, and PyTorch does not support non-writable tensors. This means writing to this tensor will result in undefined behavior. You may want to copy the array to protect its data or make it writable before converting it to a tensor. This type of warning will be suppressed for the rest of this program. (Triggered internally at ../torch/csrc/utils/tensor_numpy.cpp:206.)
docker-train-1  |   layer.weight.data = torch.from_numpy(numpy_helper.to_array(weight))
docker-train-1  | /usr/local/lib/python3.9/dist-packages/onnx2pytorch/convert/model.py:147: UserWarning: Using experimental implementation that allows 'batch_size > 1'.Batchnorm layers could potentially produce false outputs.
docker-train-1  |   warnings.warn(
docker-train-1  | [1,  2000] loss: 2.242
docker-train-1  | [1,  4000] loss: 1.972
docker-train-1  | [1,  6000] loss: 1.799
docker-train-1  | [1,  8000] loss: 1.695
docker-train-1  | [1, 10000] loss: 1.642
docker-train-1  | [1, 12000] loss: 1.581
docker-train-1  | [1, 14000] loss: 1.545
docker-train-1  | [1, 16000] loss: 1.502
docker-train-1  | [1, 18000] loss: 1.520
docker-train-1  | [1, 20000] loss: 1.471
docker-train-1  | [1, 22000] loss: 1.438
docker-train-1  | [1, 24000] loss: 1.435
docker-train-1  | [2,  2000] loss: 1.402
docker-train-1  | [2,  4000] loss: 1.358
docker-train-1  | [2,  6000] loss: 1.379
docker-train-1  | [2,  8000] loss: 1.355
...
```

## Deploy to Azure

In a more realistic scenario, this datasets will not be available in the clear to the TDC, and the TDC will be required to use a CCR for training. The following steps describe the process of sharing an encrypted dataset with TDCs and setting up a CCR in Azure for training. Please stay tuned for CCR on other cloud platforms. 

To deploy in Azure, you will need the following. 

- Docker Hub account to store container images. Alternatively, you can use pre-built images from the ```ispirt``` container registry. 
- [Azure Key Vault](https://azure.microsoft.com/en-us/products/key-vault/) to store encryption keys and implement secure key release to CCR. You can either you Azure Key Vault Premium (lower cost), or [Azure Key Vault managed HSM](https://learn.microsoft.com/en-us/azure/key-vault/managed-hsm/overview) for enhanced security. Please see instructions below on how to create and setup your AKV instance. 
- Valid Azure subscription with sufficient access to create key vault, storage accounts, storage containers, and Azure Container Instances. 

If you are using your own development environment instead of a dev container or codespaces, you will to install the following dependencies. 

- [Azure CLI](https://learn.microsoft.com/en-us/cli/azure/install-azure-cli-linux).  
- [Azure CLI Confidential containers extension](https://learn.microsoft.com/en-us/cli/azure/confcom?view=azure-cli-latest). After installing Azure CLI, you can install this extension using ```az extension add --name confcom -y```
- [Go](https://go.dev/doc/install). Follow the instructions to install Go. After installing, ensure that the PATH environment variable is set to include ```go``` runtime.
- ```jq```. You can install jq using ```sudo apt-get install -y jq```

We will be creating the following resources as part of the deployment. 

- Azure Key Vault
- Azure Storage account
- Storage containers to host encrypted datasets
- Azure Container Instances to deploy the CCR and train the model

### Push Container Images

If you wish to use your own container images, login to docker hub and push containers to your container registry. 

> **Note:** Replace `<docker-hub-registry-name>` the name of your docker hub registry name.

```bash
export CONTAINER_REGISTRY=<docker-hub-registry-name>
docker login 
./ci/push-containers.sh
cd scenarios/mnist
./ci/push-containers.sh
```

### Create Resources

Acting as the TDP, we will create a resource group, a key vault instance and storage containers to host the encrypted MNIST training dataset and encryption keys. In a real deployments, TDPs and TDCs will use their own key vault instance. However, for this sample, we will use one key vault instance to store keys for all datasets and models. 

> **Note:** At this point, automated creation of AKV managed HSMs is not supported. 

> **Note:** Replace `<resource-group-name>` and `<key-vault-endpoint>` with names of your choice. Storage account names must not container any special characters. Key vault endpoints are of the form `<key-vault-name>.vault.azure.net` (for Azure Key Vault Premium) and `<key-vault-name>.managedhsm.azure.net` for AKV managed HSM, **with no leading https**. This endpoint must be the same endpoint you used while creating the contract.

```bash
az login

export AZURE_RESOURCE_GROUP=<resource-group-name>
export AZURE_KEYVAULT_ENDPOINT=<key-vault-endpoint>
export AZURE_STORAGE_ACCOUNT_NAME=<unique-storage-account-name>
export AZURE_MNIST_CONTAINER_NAME=mnistdatacontainer
export AZURE_MODEL_CONTAINER_NAME=mnistmodelcontainer
export AZURE_OUTPUT_CONTAINER_NAME=mnistoutputcontainer

cd scenarios/covid/data
./1-create-storage-containers.sh 
./2-create-akv.sh
```

### Sign and Register Contract

Next, follow instructions [here](./../../external/contract-ledger/README.md) to sign and register a contract the contract service. The registered contract must contain reference to the dataset with matching names, keyIDs and Azure Key Vault endpoints used in this sample. A sample contract is provided [here](./contract/contract.json). After signing and registering the contract, retain the contract service URL and sequence number of the contract for the rest of this sample. 

### Import encryption keys

Next, use the following script to generate and import encryption keys into Azure Key Vault with a policy based on [policy-in-template.json](./policy/policy-in-template.json). The policy requires that the CCRs run specific containers with a specific configuration which includes the public identity of the contract service. Only CCRs that satisfy this policy will be granted access to the encryption keys. 

> **Note:** Replace `<repo-root>` with the path to and including the `depa-training` folder where the repository was cloned. 

```bash
export CONTRACT_SERVICE_URL=<contract-service-url>
export TOOLS_HOME=<repo-root>/external/confidential-sidecar-containers/tools
./3-import-keys.sh
```

The generated keys are available as files with the extension `.bin`. 

### Encrypt Dataset and Model

Next, encrypt the dataset and models using keys generated in the previous step. 

```bash
cd scenarios/mnist/data
./4-encrypt-data.sh
```

This step will generate three encrypted file system images (with extension `.img`), one for the dataset, one encrypted file system image containing the model, and one image where the trained model will be stored.

### Upload Datasets

Now upload encrypted datasets to Azure storage containers.

```bash
./5-upload-encrypted-data.sh
```

### Deploy CCR in Azure

Acting as a TDC, use the following script to deploy the CCR using Confidential Containers on Azure Container Instances. 

> **Note:** Replace `<contract-sequence-number>` with the sequence number of the contract registered with the contract service. 

```bash
cd scenarios/covid/deployment/aci
./deploy.sh -c <contract-sequence-number> -m ../../config/model_config.json -q ../../config/query_config.json
```

This script will deploy the container images from your container registry, including the encrypted filesystem sidecar. The sidecar will generate an SEV-SNP attestation report, generate an attestation token using the Microsoft Azure Attestation (MAA) service, retrieve dataset, model and output encryption keys from the TDP and TDC's Azure Key Vault, train the model, and save the resulting model into TDC's output filesystem image, which the TDC can later decrypt. 

Once the deployment is complete, you can obtain logs from the CCR using the following commands. Note there may be some delay in getting the logs are deployment is complete. 

```bash
# Obtain logs from the training container
az container logs --name depa-training-covid --resource-group $AZURE_RESOURCE_GROUP --container-name depa-training

# Obtain logs from the encrypted filesystem sidecar
az container logs --name depa-training-covid --resource-group $AZURE_RESOURCE_GROUP --container-name encrypted-storage-sidecar
```

### Download and decrypt trained model

You can download and decrypt the trained model using the following script. 

```bash
cd scenarios/covid/data
./6-download-decrypt-model.sh
```

The trained model is available in `output` folder. 

### Clean-up

You can use the following command to delete the resource group and clean-up all resources used in the demo. 

```bash
az group delete --yes --name $AZURE_RESOURCE_GROUP
```
