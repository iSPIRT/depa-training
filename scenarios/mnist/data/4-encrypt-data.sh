#!/bin/bash

./generatefs.sh -d mnist_1/preprocessed -k mnist_1_key.bin -i mnist_1.img
./generatefs.sh -d mnist_2/preprocessed -k mnist_2_key.bin -i mnist_2.img
./generatefs.sh -d mnist_3/preprocessed -k mnist_3_key.bin -i mnist_3.img
./generatefs.sh -d modeller/model -k modelkey.bin -i model.img
mkdir -p output
./generatefs.sh -d output -k outputkey.bin -i output.img
