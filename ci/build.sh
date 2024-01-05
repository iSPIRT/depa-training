#!/bin/bash

# Build pytrain
pushd src/train
python3 setup.py bdist_wheel
popd

# Build training container
docker build -f ci/Dockerfile.train src -t depa-training:latest

# Build encrypted filesystem sidecar
pushd external/confidential-sidecar-containers
./buildall.sh
popd

pushd external/contract-ledger/pyscitt
python3 setup.py bdist_wheel
popd

docker build -f ci/Dockerfile.encfs . -t depa-training-encfs