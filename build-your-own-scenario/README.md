# Build Your Own Scenario

## Overview

You can build and run your own unique Training scenarios by following three simple steps:

1. Define your high-level scenario configuration and generate a scenario boilerplate from it.
2. Implement the data preprocessing and model saving code for the Training Data Providers (TDPs) and Training Data Consumer (TDC) respectively.
3. Tailor the various training configuration files applicable to your scenario.

Once the scenario is ready, deploy it locally and/or inside a Confidential Clean Room (CCR) following the standard deployment steps.

## Step 1: Define and build your scenario template

Make sure to first complete the setup steps mentioned in the main [README](../README.md) file.

### Define your scenario configuration

In the [config](./config/) directory, you will find example scenario configuration files. These files define the high-level scenario configuration -- the datasets owned by each TDP along with associated control parameters (such as privacy budget), the training framework to use, the data joining method to use, and the model format to save/load the model in. 

In a similar fashion, you can define your own scenario configuration file following the generalized template below:

```json
{
  "scenario_name": "your-scenario-name",
  "tdps": [
    {
      "name": "data_provider_1",
      "datasets": [
        {
          "name": "dataset_name",
          "id": "unique/random-uuid-here",
          "privacy": true|false,
          "epsilon": 7.5|null,
          "delta": 0.00001|null
        }
      ]
    }
  ],
  "training_framework": "Train_DL|LLM_Finetune|Train_ML|Train_XGB",
  "join_type": "SparkJoin|DirectoryJoin",
  "model_format": "ONNX|Safetensors|HDF5|ccr_instantiate"
}
```

### Configuration Fields

- **scenario_name**: Unique name for your scenario (used for directory creation)
- **tdps**: Array of Training Data Providers (TDPs)
  - **name**: TDP identifier
  - **datasets**: Array of datasets brought by this TDP
    - **name**: Dataset name
    - **id**: Unique UUID for the dataset
    - **privacy**: (Optional) Boolean indicating if privacy protection is required
    - **epsilon**: (Optional) Privacy budget (ε) for differential privacy
    - **delta**: (Optional) Privacy parameter (δ) for differential privacy
- **training_framework**: Training framework to use among available DEPA-Training options:
  - `Train_DL`: Deep learning training
  - `LLM_Finetune`: Fine-tuning of LLMs
  - `Train_ML`: Classical machine learning training
  - `Train_XGB`: XGBoost training
- **join_type**: Data joining method to use among available DEPA-Training options:
  - `SparkJoin`: Spark-based data joining
  - `DirectoryJoin`: Directory-based data joining
- **model_format**: Format of model brought by the TDC for training
  - `ONNX`: Open Neural Network Exchange format
  - `Safetensors`: Safetensors format
  - `HDF5`: HDF5 format
  - `ccr_instantiate`: Model is created inside the CCR (no file needed)

### Generate your scenario directory

Generate a scenario directory from your scenario configuration file, by running the `build-scenario.sh` script as follows:

```bash
./build-scenario.sh <path-to-scenario.json> [--force]
```

- `<path-to-scenario.json>`: Path to your scenario configuration JSON file
- `--force`: Optional flag to overwrite existing scenario directories

Example:

```bash
./build-scenario.sh config/credit-risk.json
```

### Generated Scenario Directory Structure

```
scenarios/your-scenario-name/
├── ci/                         
│   ├── Dockerfile.*            # Dockerfiles for each TDP to prepare datasets
│   ├── Dockerfile.modelsave    # Dockerfiles for preparing base model (if applicable)
│   ├── build.sh                # Build script for all containers
│   ├── pull-containers.sh      # Container pulling script
│   └── push-containers.sh      # Container pushing script
├── src/                        
│   ├── preprocess_*.py         # Preprocessing scripts for each TDP
│   └── save_base_model.py      # Model saving script (if applicable)
├── contract/                    
│   └── contract.json           # Contract template
├── policy/                      
│   └── policy-in-template.json # Policy template
├── config/                     
│   ├── consolidate_pipeline.sh # Pipeline consolidation script
│   ├── dataset_config.json     # Dataset configuration
│   ├── eval_config.json        # Evaluation configuration
│   ├── join_config.json        # Data joining configuration (if applicable)
│   ├── model_config.json       # Model configuration (if applicable)
│   ├── loss_config.json        # Loss function configuration (DL only)
│   └── templates/              # Configuration templates for training pipeline
│       ├── pipeline_config_template.json
│       └── train_config_template.json
├── deployment/                 
│   ├── local/                  # Local deployment commands
│   └── azure/                  # Azure deployment commands
├── export-variables.sh         # Environment variables for deployment
├── .gitignore                  # Git ignore file
```

## Step 2: Implement the data preprocessing and model saving code

Prior to training, the Training Data Providers (TDPs) and Training Data Consumer (TDC) need to prepare their datasets and models respectively.

### Data preprocessing

The folder ```scenarios/your-scenario-name/src``` contains boilerplate scripts for pre-processing the datasets. Acting as a Training Data Provider (TDP), prepare your datasets by modifying the scripts according to your requirements.

Corresponding Dockerfiles are also provided in the ```scenarios/your-scenario-name/ci``` directory. Modify the Dockerfiles to install the dependencies for your preprocessing scripts.

### Model saving

If the TDC intends to bring a base model file for training, a boilerplate script for saving the model is provided in the ```scenarios/your-scenario-name/src``` directory. Acting as a TDC, modify the script to save your model in an appropriate format.

A corresponding Dockerfile is provided in the ```scenarios/your-scenario-name/ci``` directory. Modify the Dockerfile to install the dependencies for your model saving script.

## Step 3: Tailor the training configuration

The folder ```scenarios/your-scenario-name/config``` contains the training configuration files for the different training frameworks. Below is a list of configuration files that you can modify to suit your training requirements:

- `train_config_template.json`: Training configuration template
- `join_config.json`: Data joining configuration (applicable only if joining multiple datasets)
- `dataset_config.json`: Dataset configuration
- `model_config.json`: Model configuration (applicable for formats other than ONNX)
- `loss_config.json`: Loss function configuration (applicable only for DL scenarios)
- `eval_config.json`: Trained model validation configuration

## Step 4: Deploy your scenario

Now that you have the full scenario ready, you can deploy it following the same steps as the example scenarios:

### Build scenario container images

```bash
export SCENARIO=your-scenario-name
export REPO_ROOT="$(git rev-parse --show-toplevel)"
cd $REPO_ROOT/scenarios/$SCENARIO
./ci/build.sh
```

### Deploy locally

Assuming you have cleartext access to all the datasets, you can train the model _locally_ as follows:

```bash
cd $REPO_ROOT/scenarios/$SCENARIO/deployment/local
./preprocess.sh
./save-model.sh
./train.sh
```

### Deploy on CCR

Once the training scenario executes successfully in the local environment, you can train the model inside a _Confidential Clean Room (CCR)_ as follows. This reference implementation assumes Azure as the cloud platform. Stay tuned for CCR on other cloud platforms.

#### 1. Set up environment variables

Set up the necessary environment variables for your deployment in the ```scenarios/your-scenario-name/export-variables.sh``` file and run it. This will set the environment variables in the current terminal.
```bash
cd $REPO_ROOT/scenarios/$SCENARIO
./export-variables.sh
source export-variables.sh
```

#### 2. Create resources

```bash
cd $REPO_ROOT/scenarios/$SCENARIO/deployment/azure
./1-create-storage-containers.sh
./2-create-akv.sh
```

#### 3. Contract signing

Follow the instructions in the [contract-ledger](https://github.com/kapilvgit/contract-ledger/blob/main/README.md) repository for contract signing, using your scenario's contract template in `/scenarios/$SCENARIO/contract/contract.json`.

Once the contract is signed, export the contract sequence number as an environment variable in the same terminal where you set the environment variables for the deployment.

```bash
export CONTRACT_SEQ_NO=<contract-sequence-number>
```

#### 4. Data encryption and upload

```bash
cd $REPO_ROOT/scenarios/$SCENARIO/deployment/azure
./3-import-keys.sh
./4-encrypt-data.sh
./5-upload-encrypted-data.sh
```

#### 5. Deploy CCR

```bash
./deploy.sh -c $CONTRACT_SEQ_NO -p ../../config/pipeline_config.json
```

#### 6. Monitor container logs

```bash
az container logs \
  --name "depa-training-$SCENARIO" \
  --resource-group "$AZURE_RESOURCE_GROUP" \
  --container-name depa-training
```

You will know training has completed when the logs print "CCR Training complete!".

#### 7. Download and decrypt model

```bash
./6-download-decrypt-model.sh
```

The outputs will be saved to the ```scenarios/your-scenario-name/modeller/output``` directory.

## Contribute

Have a scenario that you think would be useful for others? Raise a Pull Request to contribute to the DEPA-Training project, following the [contribution guidelines](../CONTRIBUTING.md). Ensure that no personal/proprietary data, credentials or information is included in the code you submit.
