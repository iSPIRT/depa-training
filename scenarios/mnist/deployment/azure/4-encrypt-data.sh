#!/bin/bash

DATADIR=$REPO_ROOT/scenarios/$SCENARIO/data
MODELDIR=$REPO_ROOT/scenarios/$SCENARIO/modeller

./generatefs.sh -d $DATADIR/preprocessed -k $DATADIR/mnist_key.bin -i $DATADIR/mnist.img
./generatefs.sh -d $MODELDIR/models -k $MODELDIR/model_key.bin -i $MODELDIR/model.img

rm -rf $MODELDIR/output
mkdir -p $MODELDIR/output
./generatefs.sh -d $MODELDIR/output -k $MODELDIR/output_key.bin -i $MODELDIR/output.img