#!/bin/bash

./generatefs.sh -d icmr/preprocessed -k icmrkey.bin -i icmr.img
./generatefs.sh -d cowin/preprocessed -k cowinkey.bin -i cowin.img
./generatefs.sh -d index/preprocessed -k indexkey.bin -i index.img
./generatefs.sh -d modeller/model -k modelkey.bin -i model.img
mkdir -p output
./generatefs.sh -d output -k outputkey.bin -i output.img