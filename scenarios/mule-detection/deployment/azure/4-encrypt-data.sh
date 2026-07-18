#!/bin/bash

DATADIR=$REPO_ROOT/scenarios/$SCENARIO/data
MODELDIR=$REPO_ROOT/scenarios/$SCENARIO/modeller

./generatefs.sh -d $DATADIR/bank_1/preprocessed -k $DATADIR/bank_1_key.bin -i $DATADIR/bank_1.img
./generatefs.sh -d $DATADIR/bank_2/preprocessed -k $DATADIR/bank_2_key.bin -i $DATADIR/bank_2.img
./generatefs.sh -d $DATADIR/bank_3/preprocessed -k $DATADIR/bank_3_key.bin -i $DATADIR/bank_3.img

sudo rm -rf $MODELDIR/output
mkdir -p $MODELDIR/output
./generatefs.sh -d $MODELDIR/output -k $MODELDIR/output_key.bin -i $MODELDIR/output.img
