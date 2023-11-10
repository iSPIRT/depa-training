# DEPA for Training

The [DEPA for Training](https://depa.world) is a techno-legal framework that enables privacy-preserving sharing of bulk, de-identified datasets for large scale analytics and training. This repository contains a reference implementation of [Confidential Clean Rooms](https://depa.world/training/confidential_clean_room_design), which together with the [Contract Service](https://github.com/kapilvgit/contract-ledger/tree/main), forms the basis of this framework. The repository also includes a [sample training scenario](./scenarios/covid/README.md) that can be deployed using the DEPA Training Framework. The reference implementation is provided on an As-Is basis. It is work-in-progress and should not be used in production.

# Getting Started

Clone this repo as follows, and follow [instructions](./scenarios/covid/README.md) to deploy a sample CCR. 

```bash
git clone --recursive http://github.com/iSPIRT/depa-training
```

You can also use Github codespaces using the [dev container](./.devcontainer/devcontainer.json) to create a development environment. Please ensure you allocate at least 64GB disk space in your codespace. Also, run the following command in the Codespace to update submodules.

```bash
git submodule update --init --recursive
```