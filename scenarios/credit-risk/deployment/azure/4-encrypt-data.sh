#!/bin/bash

DATADIR=$REPO_ROOT/scenarios/$SCENARIO/data
MODELDIR=$REPO_ROOT/scenarios/$SCENARIO/modeller

./generatefs.sh -d $DATADIR/bank_a/preprocessed -k $DATADIR/bank_a_key.bin -i $DATADIR/bank_a.img
./generatefs.sh -d $DATADIR/bank_b/preprocessed -k $DATADIR/bank_b_key.bin -i $DATADIR/bank_b.img
./generatefs.sh -d $DATADIR/bureau/preprocessed -k $DATADIR/bureau_key.bin -i $DATADIR/bureau.img
./generatefs.sh -d $DATADIR/fintech/preprocessed -k $DATADIR/fintech_key.bin -i $DATADIR/fintech.img

sudo rm -rf $MODELDIR/output
mkdir -p $MODELDIR/output
./generatefs.sh -d $MODELDIR/output -k $MODELDIR/output_key.bin -i $MODELDIR/output.img