# Mule Account Detection

## Scenario Type

| Scenario name | Scenario type | Task type | Privacy | No. of TDPs* | Data type (format) | Model type (format) | Join type (No. of datasets) |
|---|---|---|---|---:|---|---|---|
| Mule Account Detection | Training - Classical ML | Binary Classification | Differentially Private | 3 | Derived tabular data (Parquet) | XGBoost (JSON) | Horizontal (3) |

---

## Scenario Description

This scenario trains a differentially private **XGBoost** model to identify
accounts that exhibit mule-account behaviour. Three banks act as Training Data
Providers (TDPs), each supplying an account-level feature table. The tables are
combined with a horizontal `UNION ALL` inside the Confidential Clean Room
(CCR), where the shared binary classifier is trained. A consortium-selected
bank or operator acts as the Training Data Consumer (TDC), and the CCR Provider
(CCRP) operates the attested environment.

The sample uses the public [PaySim](https://www.kaggle.com/datasets/ealaxi/paysim1)
transaction simulator. PaySim is synthetic, and the `is_mule` field produced by
this sample is a proxy label rather than a confirmed investigation outcome; a
production deployment must use institution-approved features and historical
labels.

Contributed by: Aonami Technologies

See the [technical writeup](./TECHNICAL_WRITEUP.md) for the feature formulas,
architecture, privacy analysis, and implementation details.

The end-to-end training pipeline consists of the following phases:

1. Data pre-processing
2. Data packaging, encryption, and upload
3. Encryption-key import with key-release policies
4. CCR deployment and model training
5. Trained-model decryption

## Build container images

Build the core DEPA Training images and the three bank pre-processing images
from the repository root:

```bash
export SCENARIO=mule-detection
export REPO_ROOT="$(git rev-parse --show-toplevel)"

cd "$REPO_ROOT"
./ci/build.sh

cd "$REPO_ROOT/scenarios/$SCENARIO"
./ci/build.sh
```

The scenario build creates:

- `preprocess-bank-1:latest`
- `preprocess-bank-2:latest`
- `preprocess-bank-3:latest`

Alternatively, pull scenario images that have already been published to your
container registry:

```bash
export CONTAINER_REGISTRY=<container-registry-name>
cd "$REPO_ROOT/scenarios/$SCENARIO"
./ci/pull-containers.sh
```

## Data pre-processing

Set the Kaggle credentials available from your Kaggle account settings. Keep
these credentials in environment variables or an ignored local file; do not
commit them.

```bash
export KAGGLE_USERNAME=<kaggle-username>
export KAGGLE_KEY=<kaggle-key>
```

Run the three bank pre-processors:

```bash
cd "$REPO_ROOT/scenarios/$SCENARIO/deployment/local"
./preprocess.sh
```

The pre-processors create one Parquet file per bank under
`scenarios/mule-detection/data/`. Each file has the following schema:

```text
account_age_days, vel_24, vel_48, vel_72,
fan_in, fan_out, dormancy_ratio, is_mule
```

Raw transactions, counterparty identifiers, and `account_id` are not included
in the provider Parquet output. The sample partitions PaySim accounts among the
three providers only to emulate separate bank datasets; it does not represent
cross-bank access to production ledgers.

## Deploy locally

If the three pre-processed datasets are available locally, train the model as
follows:

```bash
cd "$REPO_ROOT/scenarios/$SCENARIO/deployment/local"
./train.sh
```

The script consolidates the files in the [config](./config/) directory,
combines the three provider tables, and runs XGBoost training. The trained model
and evaluation artifacts are saved under
`scenarios/mule-detection/modeller/output/`. Results from the synthetic fixture
are pipeline diagnostics, not evidence of production fraud-detection
performance.

## Deploy on CCR

In a consortium deployment, each TDP encrypts and uploads its own feature table.
The TDC then deploys the training pipeline to an Azure CCR, where the datasets
are joined and the model is trained without exposing the cleartext inputs to
the TDC.

The Azure workflow requires:

- Docker and access to a container registry
- Azure CLI with the confidential-containers extension
- `jq`, `envsubst`, Go, `cryptsetup`, and `bc`
- an Azure subscription permitted to create a resource group, storage account,
  Key Vault Premium instance, and confidential Azure Container Instance
- three distinct bank `did:web` identities, a TDC DID, and a CCRP DID
- a consortium-selected contract service reachable through a non-loopback
  HTTPS URL

### 1. Push Container Images

If the required images have not yet been published, authenticate with your
registry and push the core and scenario images:

```bash
export CONTAINER_REGISTRY=<container-registry-name>
docker login "$CONTAINER_REGISTRY"

cd "$REPO_ROOT"
./ci/push-containers.sh

cd "$REPO_ROOT/scenarios/$SCENARIO"
./ci/push-containers.sh
```

### 2. Create Resources

Authenticate with Azure, edit
[`export-variables.sh`](./export-variables.sh), and source it:

```bash
az login
cd "$REPO_ROOT/scenarios/$SCENARIO"
source ./export-variables.sh
```

Set the subscription, resource group, Key Vault endpoint, storage-account name,
participant DIDs, and contract-service URL before continuing. The Key Vault
endpoint must use the form `https://<name>.vault.azure.net`, and
`CONTRACT_SERVICE_URL` must be a reachable non-loopback HTTPS endpoint.

Create the storage containers and Key Vault:

```bash
cd "$REPO_ROOT/scenarios/$SCENARIO/deployment/azure"
./1-create-storage-containers.sh
./2-create-akv.sh
```

### 3. Contract Signing

Render the infrastructure-specific unsigned contract:

```bash
cd "$REPO_ROOT/scenarios/$SCENARIO"
./contract/prepare-contract.sh /tmp/mule-detection-contract.json
```

The renderer maps each dataset to its bank DID and requires three distinct bank
`did:web` identities plus the TDC and CCRP DIDs. Complete the official
[contract-service signing flow](https://github.com/iSPIRT/contract-service/blob/main/demo/contract/README.md):
each bank signs and registers the latest contract in sequence, followed by the
TDC. Retain the final registered contract sequence for CCR deployment. Do not
substitute local Git identities or a source-code override for participant
signatures.

### 4. Data Encryption and Upload

Import the attestation-bound keys, create the encrypted filesystem images, and
upload them to Azure Storage:

```bash
cd "$REPO_ROOT/scenarios/$SCENARIO/deployment/azure"
./3-import-keys.sh
./4-encrypt-data.sh
./5-upload-encrypted-data.sh
```

This creates separate encrypted images for the three provider tables and a
writable encrypted image for model output.

### 5. CCR Deployment

Consolidate the pipeline configuration and deploy the CCR using the final
registered contract sequence:

```bash
cd "$REPO_ROOT/scenarios/$SCENARIO/config"
./consolidate_pipeline.sh

export CONTRACT_SEQ_NO=<registered-contract-sequence>
cd "$REPO_ROOT/scenarios/$SCENARIO/deployment/azure"
./deploy.sh -c "$CONTRACT_SEQ_NO" -p ../../config/pipeline_config.json
```

The contract-service URL, Azure resources, participant identities, and pipeline
configuration must match the values used to generate the key-release policy and
signed contract.

### 6. Monitor Container Logs

Creating the Azure Container Instance does not mean training has completed.
Monitor the training container until it prints `CCR Training complete!`:

```bash
az container logs \
  --name "depa-training-$SCENARIO" \
  --resource-group "$AZURE_RESOURCE_GROUP" \
  --container-name depa-training
```

If training fails, inspect the encrypted-storage sidecar logs:

```bash
az container logs \
  --name "depa-training-$SCENARIO" \
  --resource-group "$AZURE_RESOURCE_GROUP" \
  --container-name encrypted-storage-sidecar
```

### 7. Download and Decrypt Model

After training completes, download and decrypt the model and evaluation
artifacts:

```bash
cd "$REPO_ROOT/scenarios/$SCENARIO/deployment/azure"
./6-download-decrypt-model.sh
```

The decrypted files are saved under
`scenarios/mule-detection/modeller/output/`.

## Clean-up

Delete the demonstration resource group when it is no longer required:

```bash
az group delete --yes --name "$AZURE_RESOURCE_GROUP"
```

Revoke temporary access tokens and remove generated credentials, keys,
encrypted images, datasets, and model outputs according to your organization's
retention policy. These artifacts must not be committed to Git.

## References

- [DEPA Training Framework](https://depa.world/training/depa_training_framework/)
- [iSPIRT DEPA Training contract service](https://github.com/iSPIRT/contract-service)
- [PaySim dataset](https://www.kaggle.com/datasets/ealaxi/paysim1)
- [XGBoost documentation](https://xgboost.readthedocs.io/en/stable/)
