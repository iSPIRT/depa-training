#!/bin/bash

DATADIR=$REPO_ROOT/scenarios/$SCENARIO/data
MODELDIR=$REPO_ROOT/scenarios/$SCENARIO/modeller

./generatefs.sh -d $DATADIR/oral_cyto_A/preprocessed -k $DATADIR/oral_cyto_A_key.bin -i $DATADIR/oral_cyto_A.img
./generatefs.sh -d $DATADIR/oral_cyto_B/preprocessed -k $DATADIR/oral_cyto_B_key.bin -i $DATADIR/oral_cyto_B.img

./generatefs.sh -d $MODELDIR/models -k $MODELDIR/model_key.bin -i $MODELDIR/model.img

sudo rm -rf $MODELDIR/output
mkdir -p $MODELDIR/output
./generatefs.sh -d $MODELDIR/output -k $MODELDIR/output_key.bin -i $MODELDIR/output.img
