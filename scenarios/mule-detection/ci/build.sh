#!/bin/bash
docker build -f ci/Dockerfile.bank1 src -t preprocess-bank-1:latest
docker build -f ci/Dockerfile.bank2 src -t preprocess-bank-2:latest
docker build -f ci/Dockerfile.bank3 src -t preprocess-bank-3:latest
