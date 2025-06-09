# LLM Fine-tuning with Differential Privacy

This scenario demonstrates how a Large Language Model (LLM) can be fine-tuned for medical question answering using the join of multiple (potentially PII-sensitive) datasets. The Training Data Consumer (TDC) building the model gets into a contractual agreement with multiple Training Data Providers (TDPs), and the model is fine-tuned on the joined datasets in a data-blind manner within the CCR, maintaining privacy guarantees using differential privacy. For demonstration purposes, this scenario uses open-source models and datasets from HuggingFace.

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
cd scenarios/llm-finetune
./ci/build.sh
```

This script builds the following container images:

- `preprocess-icmr, preprocess-cowin, preprocess-index`: Containers that pre-process the text datasets.
- `ccr-model-save`: Container that saves the base model to be fine-tuned.

## Data pre-processing and de-identification

Acting as TDPs for each dataset, run the following scripts to de-identify the datasets:

```bash
cd scenarios/llm-finetune/deployment/docker
./preprocess.sh
```

This script performs pre-processing of the text datasets before the training process.

## Prepare model for training

Next, acting as a TDC, load and save a sample model using the following script:

```bash
./save-model.sh
```

This script will save the base model within `scenarios/llm-finetune/model/`.

## Deploy locally

Assuming you have cleartext access to all the datasets, you can fine-tune the model as follows:

```bash
./train.sh
```

The script joins the datasets and fine-tunes the model using a pipeline configuration defined in [pipeline_config.json](./config/pipeline_config.json). The fine-tuning process uses:

- LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning
- Differential Privacy via Opacus
- Weight quantization for memory efficiency

If all goes well, you should see training progress output similar to:

```
Epoch: 1 | Step: 50 | Train loss: 2.342
Epoch: 1 | Step: 100 | Train loss: 2.123
...
Epoch 1 completed. Average loss: 2.156
```

The fine-tuned model will be saved under the folder `/scenarios/llm-finetune/output`.

## Deploy on CCR
