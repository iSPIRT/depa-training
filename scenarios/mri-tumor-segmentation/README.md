# MRI Tumor Segmentation with Differential Privacy

This scenario demonstrates how a deep learning model can be trained for MRI Tumor Segmentation using the join of multiple (potentially PII-sensitive) medical imaging datasets. The Training Data Consumer (TDC) building the model gets into a contractual agreement with multiple Training Data Providers (TDPs) having annotated MRI data, and the model is trained on the joined datasets in a data-blind manner within the CCR, maintaining privacy guarantees (as per need) using differential privacy. For demonstration purpose, this scenario uses annotated MRI data made available through the BRaTS 2020 challenge, and a custom UNet architecture model for segmentation.

The end-to-end training pipeline consists of the following phases:

1. Data pre-processing
2. Data packaging, encryption and upload
3. Model packaging, encryption and upload
4. Encryption key import with key release policies
5. Deployment and execution of CCR
6. Model decryption

## Build container images

Build container images required for this sample as follows:

```bash
cd ~/depa-training/scenarios/mri-tumor-segmentation
./ci/build.sh
```

This script builds the following container images:

- `preprocess-brats-a, preprocess-brats-b, preprocess-brats-c`: Containers that pre-process the individual MRI datasets
- `brats-model-save`: Container that saves the base model to be trained

## Data pre-processing

For ease of execution, the individual preprocessed datasets are already made available in the repo under `scenarios/mri-tumor-segmentation/data`. If you wish to pre-process the datasets yourself (in this case, extract 2D slices from the 3D MRI NIfTI volumes), acting as TDPs for each dataset, run the following scripts:

```bash
cd ~/depa-training/scenarios/mri-tumor-segmentation/deployment/docker
./preprocess.sh
```

This script performs pre-processing of the MRI datasets before the training process, including:

- Slice extraction (2D slices from 3D volumes)
- Image normalization
- Data augmentation (optional)

## Prepare model for training

Next, acting as a TDC, load and save a sample model using the following script:

```bash
cd ~/depa-training/scenarios/mri-tumor-segmentation/deployment/docker
./save-model.sh
```

This script will save the base model within `scenarios/mri-tumor-segmentation/model/`.

## Deploy locally

Assuming you have cleartext access to all the datasets, you can train the model as follows:

```bash
cd ~/depa-training/scenarios/mri-tumor-segmentation/deployment/docker
./train.sh
```

The script joins the datasets and trains the model using a pipeline configuration defined in [pipeline_config.json](./config/pipeline_config.json). The training process uses:

- U-Net architecture for medical image segmentation
- Differential Privacy via Opacus
- Data augmentation (optional) for improved generalization

If all goes well, you should see training progress output similar to:

```
Epoch: 1 | Step: 50 | Train loss: 0.342 | Dice score: 0.823
Epoch: 1 | Step: 100 | Train loss: 0.223 | Dice score: 0.856
...
Epoch 1 completed. Average loss: 0.256 | Average Dice: 0.845
```

The trained model along with sample validation set outputs will be saved under the folder `~/depa-training/scenarios/mri-tumor-segmentation/output`.

## Deploy on CCR

Instructions for executing the DEPA-Training MRI Tumor Segmentation scenario on Azure Container Instances (ACI), with confidential computing, key release policies, and contract-based access control.

**Important:** Before proceeding with contract setup and signing, ensure the `contract.json` template used during contract signing is correctly updated for the MRI-Tumor-Segmentation scenario. **Replace it with the one from `scenarios/mri-tumor-segmentation/contract/contract.json`.**

Follow these steps to deploy and run the DEPA-Training solution on ACI.

### 1. Azure and Docker Login

```bash
az login
docker login
```

---

### 2. Export Environment Variables

Set up the necessary environment variables for your deployment.

```bash
export CONTAINER_REGISTRY=sarang10
export AZURE_LOCATION=southindia
export AZURE_SUBSCRIPTION_ID=2a5f1e30-b076-4cb2-9235-2036241dedf0
export AZURE_RESOURCE_GROUP=depa-train-ccr-demo
export AZURE_KEYVAULT_ENDPOINT=depa-train-ccr-dev.vault.azure.net
export AZURE_STORAGE_ACCOUNT_NAME=depatrainccrdev
export AZURE_BRATS_A_CONTAINER_NAME=bratsacontainer
export AZURE_BRATS_B_CONTAINER_NAME=bratsbcontainer
export AZURE_BRATS_C_CONTAINER_NAME=bratsccontainer
export AZURE_BRATS_D_CONTAINER_NAME=bratsdcontainer
export AZURE_MODEL_CONTAINER_NAME=bratsmodelcontainer
export AZURE_OUTPUT_CONTAINER_NAME=bratsoutputcontainer
export CONTRACT_SERVICE_URL=[https://depa-contract-service.southindia.cloudapp.azure.com:8000](https://depa-contract-service.southindia.cloudapp.azure.com:8000)
export TOOLS_HOME=/home/depa-train-ccr-dev/depa-training/external/confidential-sidecar-containers/tools
```

---

**Important:**

- The value for **`AZURE_KEYVAULT_ENDPOINT`** must precisely match the **`TDP_KEYVAULT`** environment variable during contract service execution. Any mismatch will lead to execution failure.

- Verify that **`AZURE_STORAGE_ACCOUNT_NAME`** corresponds exactly to the storage account name specified in the URLs of your `demo/contract/contract.json` file during contract signing (e.g., "depatrainccrdev" in the "URL" field of the "datasets" section in `contract.json`).

- Confirm that your Azure container names (e.g., "bratsacontainer", "bratsbcontainer", "bratsccontainer", "bratsdcontainer") are identical to those listed in the `contract.json` URLs (e.g., "https://depatrainccrdev.blob.core.windows.net/bratsacontainer/data.img"). These names should be **all lowercase and contain no special characters.**

---

### 3\. Data Preparation and Key Management

Navigate to the data directory and execute the scripts for storage container creation, Azure Key Vault setup, key import, and data encryption and upload to Azure Blob Storage.

```bash
cd ~/depa-training/scenarios/mri-tumor-segmentation/data

./1-create-storage-containers.sh

./2-create-akv.sh

./3-import-keys.sh

./4-encrypt-data.sh

./5-upload-encrypted-data.sh
```

### 4\. ACI Deployment

Move to the ACI deployment directory and run the deployment script.

```bash
cd ../deployment/aci

./deploy.sh -c <contract-sequence-number> -p ../../config/pipeline_config.json
```

**Note:** The `<contract-sequence-number>` is the number of the contract that was signed with the TDPs (see contract service instructions)

**Example:**

```bash
./deploy.sh -c 15 -p ../../config/pipeline_config.json
```

**Note:** Even if the script indicates "deployment complete", the training process might still be ongoing. Monitor the container logs to track progress until training is complete.

### 5\. Monitor Container Logs

Use the following commands to monitor the logs of the deployed containers. You might have to repeatedly poll this command to monitor the training progress:

```bash
az container logs --name depa-training-brats --resource-group $AZURE_RESOURCE_GROUP --container-name depa-training
```

In case training fails, you might want to monitor the logs of the encrypted storage sidecar container to see if the encryption process completed successfully:

```bash
az container logs --name depa-training-brats --resource-group $AZURE_RESOURCE_GROUP --container-name encrypted-storage-sidecar
```

### 6\. Download and Decrypt Model

Once training is complete, navigate back to the data directory to download and decrypt the trained model.

```bash
cd ../../data

./6-download-decrypt-model.sh
```

**Verification:** If the `scenarios/mri-tumor-segmentation/output` folder now contains a fresh version of `trained_model.pth`, this indicates successful end-to-end deployment of DEPA-Training on ACI.

To check if the trained model is fresh, you can run the following command:

```bash
stat ~/depa-training/scenarios/mri-tumor-segmentation/output/trained_model.pth
```

---

<br>
<br>

### Troubleshooting

(work in progress)

Here are some common issues (and their solutions for some of them):

**Issue:** `ls: cannot access '/mnt/remote/config/pipeline_config.json': No such file or directory`

**Command:**

```bash
az container logs --name depa-training-mnist --resource-group $AZURE_RESOURCE_GROUP --container-name depa-training
```

**Output:**

```
ls: cannot access '/mnt/remote/config/pipeline_config.json': No such file or directory
```

**Reason:** (Not provided in notes, but typically indicates a missing file or incorrect path for `pipeline_config.json` within the container.)
**Fix:** (No fix provided in notes.)

### Issue: `DeploymentFailed` during `deploy.sh`

**Command:**

```bash
./deploy.sh -c <contract-sequence-number> -p ../../config/pipeline_config.json
```

**Output:**

```json
{
  "status": "Failed",
  "error": {
    "code": "DeploymentFailed",
    "target": "/subscriptions/2a5f1e30-b076-4cb2-9235-2036241dedf0/resourceGroups/depa-train-ccr-demo/providers/Microsoft.Resources/deployments/arm-template",
    "message": "At least one resource deployment operation failed. Please list deployment operations for details. Please see [https://aka.ms/arm-deployment-operations](https://aka.ms/arm-deployment-operations) for usage details.",
    "details": [
      {
        "code": "ResourceDeploymentFailure",
        "target": "/subscriptions/2a5f1e30-b076-4cb2-9235-2036241dedf0/resourceGroups/depa-train-ccr-demo/providers/Microsoft.ContainerInstance/containerGroups/depa-training-brats",
        "message": "The resource write operation failed to complete successfully, because it reached terminal provisioning state 'Failed'.",
        "details": [{}]
      }
    ]
  }
}
```

**Reason:** The `run.sh` script inside the `depa-training-brats` container did not have execution permission.

**Fix (Update: Below not solving the problem):**

1.  Navigate to the source directory:
    ```bash
    cd ~/depa-training/src/train
    ```
2.  Check file permissions for `run.sh` and `setup.py`:
    ```bash
    ls -l
    ```
3.  If execution (`x`) isn't mentioned, run:
    ```bash
    chmod +x run.sh
    ```
4.  Then rebuild and push containers:
    ```bash
    cd ~/depa-training
    ./ci/build.sh
    export CONTAINER_REGISTRY=sarang10
    ./ci/push-containers.sh
    ```

### Issue: `deploy.sh` running "Running" (endlessly)

**Command:**

```bash
./deploy.sh -c <contract-sequence-number> -p ../../config/pipeline_config.json
```

**Output:**

```
“Running” (endlessly)
```

Running `az container show --resource-group depa-train-ccr-demo --name depa-training-brats --output json` shows instanceView state "pending" and provisioningstate "Creating".

**Reason:** Error with container configurations.

**Fix:** Ensure that in `deployment/aci/encrypted-filesystem-config-template.json` there are as many dictionaries as there are mounting paths. For example, if there are 4 TDP containers and 2 TDC containers, there must be 6 dictionaries in total, with the last one (Output container) having `read_write=true`.

### Issue: `ContainerGroupTransitioning`

**Command:** (Any deployment command after a previous failure)

**Output:**

```json
{
  "status": "Failed",
  "error": {
    "code": "DeploymentFailed",
    "target": "/subscriptions/2a5f1e30-b076-4cb2-9235-2036241dedf0/resourceGroups/depa-train-ccr-demo/providers/Microsoft.Resources/deployments/arm-template",
    "message": "At least one resource deployment operation failed. Please list deployment operations for details. Please see [https://aka.ms/arm-deployment-operations](https://aka.ms/arm-deployment-operations) for usage details.",
    "details": [
      {
        "code": "ContainerGroupTransitioning",
        "message": "The container group 'depa-training-brats' is still transitioning, please retry later."
      }
    ]
  }
}
```

**Reason:** You may have an existing Azure container group (e.g., `depa-training-brats`) which is broken or stuck in a transitioning state.

**Fix:** Go to the Azure portal, find the problematic container group (e.g., `depa-training-brats`), stop and delete it. Then run the deploy script again.

---

```

```
