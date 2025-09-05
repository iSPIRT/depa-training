#!/bin/bash

docker build -f ci/Dockerfile.bankA src -t preprocess-bank-a:latest
docker build -f ci/Dockerfile.bankB src -t preprocess-bank-b:latest
docker build -f ci/Dockerfile.bureau src -t preprocess-bureau:latest
docker build -f ci/Dockerfile.fintech src -t preprocess-fintech:latest