#!/bin/bash

./generatefs.sh -d brats_A/preprocessed -k brats_A_key.bin -i brats_A.img
./generatefs.sh -d brats_B/preprocessed -k brats_B_key.bin -i brats_B.img
./generatefs.sh -d brats_C/preprocessed -k brats_C_key.bin -i brats_C.img
./generatefs.sh -d brats_D/preprocessed -k brats_D_key.bin -i brats_D.img
./generatefs.sh -d ../model -k model_key.bin -i model.img
# mkdir -p output
./generatefs.sh -d ../output -k output_key.bin -i output.img