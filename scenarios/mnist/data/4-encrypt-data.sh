#!/bin/bash

./generatefs.sh -d preprocessed -k mnistkey.bin -i mnist.img
./generatefs.sh -d model -k modelkey.bin -i model.img
mkdir -p output
./generatefs.sh -d output -k outputkey.bin -i output.img