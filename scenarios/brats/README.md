# Brain MRI Tumor Segmentation

This scenario demonstrates how a deep learning model can be trained for MRI Tumor Segmentation using the join of multiple (potentially PII-sensitive) medical imaging datasets. The Training Data Consumer (TDC) building the model gets into a contractual agreement with multiple Training Data Providers (TDPs) having annotated MRI data, and the model is trained on the joined datasets in a data-blind manner within the CCR, maintaining privacy guarantees (as per need) using differential privacy. For demonstration purpose, this scenario uses annotated MRI data made available through the BRaTS 2020 challenge, and a custom UNet architecture model for segmentation.

For this demo, we use the BraTS 2020 challenge datasets [1] [2] [3].

The end-to-end training pipeline consists of the following phases:

1. Data pre-processing
2. Packaging, encryption and upload of data and model
3. Model packaging, encryption and upload
4. Encryption key import with key release policies
5. Deployment and execution of CCR
6. Model decryption

## Build container images

Build container images required for this sample as follows:

```bash
cd ~/depa-training/scenarios/brats
./ci/build.sh
```

This script builds the following container images:

- `preprocess-brats-a, preprocess-brats-b, preprocess-brats-c`: Containers that pre-process the individual MRI datasets
- `brats-model-save`: Container that saves the base model to be trained

Alternatively, you can pull and use pre-built container images from the ispirt container registry by setting the following environment variable. Docker hub has started throttling which may effect the upload/download time, especially when images are bigger size. So, It is advisable to use other container registries. We are using Azure container registry (ACR) as shown below:

```bash
export CONTAINER_REGISTRY=depatraindevacr.azurecr.io
cd ~/depa-training/scenarios/brats
./ci/pull-containers.sh
```

## Data pre-processing

Acting as a Training Data Provider (TDP), prepare your datasets.

For ease of execution, the individual preprocessed BraTS MRI datasets are already made available in the repo under `scenarios/brats/data` as `tar.gz` files. Run the following scripts to extract them:

```bash
cd ~/depa-training/scenarios/brats/deployment/docker
./preprocess.sh
```

The datasets are saved to the [data](./data/) directory.

> Note: If you wish to pre-process the datasets yourself (in this case, extract 2D slices from the original 3D MRI NIfTI volumes and perform preprocessing and augmentation steps), uncomment and modify the preprocess scripts located in [src](./src/preprocess_brats_A.py).

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

- Convolutional U-Net model architecture for image segmentation.
- Differential Privacy to prevent reconstruction & membership inference attacks, using the Opacus library.
- PyTorch for training the model.

If all goes well, you should see training progress output similar to:

```bash
Epoch: 1 | Step: 50 | Train loss: 0.342 | Dice score: 0.823
Epoch: 1 | Step: 100 | Train loss: 0.223 | Dice score: 0.856
...
Epoch 1 completed. Average loss: 0.256 | Average Dice: 0.845
```

The trained model along with sample predictions on the validation set will be saved under the [output](./modeller/output/) directory.

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
export SCENARIO=brats
export CONTAINER_REGISTRY=depatraindevacr.azurecr.io
export AZURE_LOCATION=northeurope
export AZURE_SUBSCRIPTION_ID=<azure-subscription-id>
export AZURE_RESOURCE_GROUP=<resource-group-name>
export AZURE_KEYVAULT_ENDPOINT=<key-vault-endpoint-name>.vault.azure.net
export AZURE_STORAGE_ACCOUNT_NAME=<storage-account-name>

export AZURE_BRATS_A_CONTAINER_NAME=bratsacontainer
export AZURE_BRATS_B_CONTAINER_NAME=bratsbcontainer
export AZURE_BRATS_C_CONTAINER_NAME=bratsccontainer
export AZURE_BRATS_D_CONTAINER_NAME=bratsdcontainer
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
- `AZURE_BRATS_A_CONTAINER_NAME`
- `AZURE_BRATS_B_CONTAINER_NAME`
- `AZURE_BRATS_C_CONTAINER_NAME`
- `AZURE_BRATS_D_CONTAINER_NAME`

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

### References

[1] B. H. Menze, A. Jakab, S. Bauer, J. Kalpathy-Cramer, K. Farahani, J. Kirby, et al. "The Multimodal Brain Tumor Image Segmentation Benchmark (BRATS)", IEEE Transactions on Medical Imaging 34(10), 1993-2024 (2015) DOI: 10.1109/TMI.2014.2377694 (opens in a new window)

[2] S. Bakas, H. Akbari, A. Sotiras, M. Bilello, M. Rozycki, J.S. Kirby, et al., "Advancing The Cancer Genome Atlas glioma MRI collections with expert segmentation labels and radiomic features", Nature Scientific Data, 4:170117 (2017) DOI: 10.1038/sdata.2017.117(opens in a new window)

[3] S. Bakas, M. Reyes, A. Jakab, S. Bauer, M. Rempfler, A. Crimi, et al., "Identifying the Best Machine Learning Algorithms for Brain Tumor Segmentation, Progression Assessment, and Overall Survival Prediction in the BRATS Challenge", arXiv preprint arXiv:1811.02629 (2018)













<br>
<br>
<!-- 
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

``` -->
