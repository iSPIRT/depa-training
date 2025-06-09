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
cd scenarios/mri-tumor-segmentation
./ci/build.sh
```

This script builds the following container images:

- `preprocess-brats-a, preprocess-brats-b, preprocess-brats-c`: Containers that pre-process the individual MRI datasets
- `ccr-model-save`: Container that saves the base model to be trained

## Data pre-processing

For ease of execution, the individual preprocessed datasets are already made available in the repo under `scenarios/mri-tumor-segmentation/data`. If you wish to pre-process the datasets yourself (in this case, extract 2D slices from the 3D MRI NIfTI volumes), acting as TDPs for each dataset, run the following scripts:

```bash
cd scenarios/mri-tumor-segmentation/deployment/docker
./preprocess.sh
```

This script performs pre-processing of the MRI datasets before the training process, including:

- Slice extraction (2D slices from 3D volumes)
- Image normalization
- Data augmentation (optional)

## Prepare model for training

Next, acting as a TDC, load and save a sample model using the following script:

```bash
./save-model.sh
```

This script will save the base model within `scenarios/mri-tumor-segmentation/model/`.

## Deploy locally

Assuming you have cleartext access to all the datasets, you can train the model as follows:

```bash
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

The trained model along with sample validation set outputs will be saved under the folder `/scenarios/mri-tumor-segmentation/output`.

## Deploy on CCR
