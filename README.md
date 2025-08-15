# DEPA for Training

[DEPA for Training](https://depa.world) is a techno-legal framework that enables privacy-preserving sharing of bulk, de-identified datasets for large scale analytics and training. This repository contains a reference implementation of [Confidential Clean Rooms](https://depa.world/training/confidential_clean_room_design), which together with the [Contract Service](https://github.com/kapilvgit/contract-ledger/tree/main), forms the basis of this framework. The reference implementation is provided on an As-Is basis. It is work-in-progress and should not be used in production.

# Getting Started

## GitHub Codespaces

The simplest way to setup a development environment is using [GitHub Codespaces](https://github.com/codespaces). The repository includes a [devcontainer.json](../../.devcontainer/devcontainer.json), which customizes your codespace to install all required dependencies. Please ensure you allocate at least 8 vCPUs and 64GB disk space in your codespace. Also, run the following command in the codespace to update submodules.

```bash
git submodule update --init --recursive
```

## Local Development Environment

Alternatively, you can build and develop locally in a Linux environment (we have tested with Ubuntu 20.04 and 22.04), or Windows with WSL 2. 

Clone this repo to your local machine / virtual machine as follows. 

```bash
git clone --recursive http://github.com/iSPIRT/depa-training
cd depa-training
```

Install the below listed dependencies by running the [install-prerequisites.sh](./install-prerequisites.sh) script.

```bash
./install-prerequisites.sh
```

This script installs the following core dependencies, among others, which you can also install manually as follows.

- [docker](https://docs.docker.com/engine/install/ubuntu/) and docker-compose. After installing docker, add your user to the docker group using `sudo usermod -aG docker $USER`, and log back in to a shell. This may require a machine restart to take effect.

```bash
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo apt install docker-compose
sudo usermod -aG docker $USER
```

- Make (install using ```sudo apt-get install make```)
- Python3 (>=3.9) and pip (install using ```sudo apt install python3``` and ```sudo apt install python3-pip```) 
- [Go](https://go.dev/doc/install). (```sudo apt install golang-go```) Follow the instructions to install Go. After installing, ensure that the PATH environment variable is set to include ```go``` runtime.
- Python wheel package (install using ```pip install wheel```)

## Build CCR containers

To build your own CCR container images, use the following command from the root of the repository. 

```bash
./ci/build.sh
```

This scripts build the following containers. 

- ```depa-training```: Container with the core CCR logic for joining datasets and running differentially private training. 
- ```depa-training-encfs```: Container for loading encrypted data into the CCR. 

Alternatively, you can use pre-built container images from the ispirt repository by setting the following environment variable. Docker hub has started throttling which may effect the upload/download time, especially when images are bigger size. So, It is advisable to use other container registries, we are using azure container registry as shown below
```bash
export CONTAINER_REGISTRY=depatraindevacr.azurecr.io
./ci/pull-containers.sh
```

# Scenarios

This repository contains two samples that illustrate the kinds of scenarios DEPA for Training can support. 

| Scenario name | Scenario type | Training method | Dataset type | Join type |
|--------------|---------------|-----------------|--------------|-----------|
| [COVID-19](./scenarios/covid/README.md) | Training | Differentially Private Classification | PII tabular dataset | Horizontal |
| [BraTS](./scenarios/brats/README.md) | Training | Differentially Private Segmentation | PII image dataset | Vertical |
| [MNIST](./scenarios/mnist/README.md) | Training | Classification | Non-PII image dataset | NA (no join) |

Follow the links to build and deploy these scenarios. 

# Contributing

This project welcomes feedback and contributions. Before you start, please take a moment to review our [Contribution Guidelines](./CONTRIBUTING.md). These guidelines provide information on how to contribute, set up your development environment, and submit your changes.

We look forward to your contributions and appreciate your efforts in making DEPA Training better for everyone.
