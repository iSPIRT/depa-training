#!/bin/bash

DATADIR=$REPO_ROOT/scenarios/$SCENARIO/data
MODELDIR=$REPO_ROOT/scenarios/$SCENARIO/modeller

./generatefs.sh -d $DATADIR/icmr/preprocessed -k $DATADIR/icmr_key.bin -i $DATADIR/icmr.img
./generatefs.sh -d $DATADIR/cowin/preprocessed -k $DATADIR/cowin_key.bin -i $DATADIR/cowin.img
./generatefs.sh -d $DATADIR/index/preprocessed -k $DATADIR/index_key.bin -i $DATADIR/index.img

# NEW: Add model and custom dataset definitions to the model filesystem
cp $MODELDIR/../src/class_definitions.py $MODELDIR/models/class_definitions.py
./generatefs.sh -d $MODELDIR/models -k $MODELDIR/model_key.bin -i $MODELDIR/model.img

rm -rf $MODELDIR/output
mkdir -p $MODELDIR/output
./generatefs.sh -d $MODELDIR/output -k $MODELDIR/output_key.bin -i $MODELDIR/output.img