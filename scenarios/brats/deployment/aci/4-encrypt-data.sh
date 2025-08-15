#!/bin/bash

DATADIR=$REPO_ROOT/scenarios/$SCENARIO/data
MODELDIR=$REPO_ROOT/scenarios/$SCENARIO/modeller

./generatefs.sh -d $DATADIR/brats_A/preprocessed -k $DATADIR/brats_A_key.bin -i $DATADIR/brats_A.img
./generatefs.sh -d $DATADIR/brats_B/preprocessed -k $DATADIR/brats_B_key.bin -i $DATADIR/brats_B.img
./generatefs.sh -d $DATADIR/brats_C/preprocessed -k $DATADIR/brats_C_key.bin -i $DATADIR/brats_C.img
./generatefs.sh -d $DATADIR/brats_D/preprocessed -k $DATADIR/brats_D_key.bin -i $DATADIR/brats_D.img

# NEW: Add model and custom dataset definitions to the model filesystem
cp $MODELDIR/../src/class_definitions.py $MODELDIR/models/class_definitions.py
./generatefs.sh -d $MODELDIR/models -k $MODELDIR/model_key.bin -i $MODELDIR/model.img

rm -rf $MODELDIR/output
mkdir -p $MODELDIR/output
./generatefs.sh -d $MODELDIR/output -k $MODELDIR/output_key.bin -i $MODELDIR/output.img