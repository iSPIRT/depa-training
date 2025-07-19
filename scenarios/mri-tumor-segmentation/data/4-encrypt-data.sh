#!/bin/bash

./generatefs.sh -d brats_A/preprocessed -k brats_a_key.bin -i brats_a.img
./generatefs.sh -d brats_B/preprocessed -k brats_b_key.bin -i brats_b.img
./generatefs.sh -d brats_C/preprocessed -k brats_c_key.bin -i brats_c.img
./generatefs.sh -d brats_D/preprocessed -k brats_d_key.bin -i brats_d.img
./generatefs.sh -d ../model -k model_key.bin -i model.img
# mkdir -p output
./generatefs.sh -d ../output -k output_key.bin -i output.img