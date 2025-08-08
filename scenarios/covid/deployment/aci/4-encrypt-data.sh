#!/bin/bash

DATADIR=~/depa-training/scenarios/$SCENARIO/data
MODELDIR=~/depa-training/scenarios/$SCENARIO/modeller

./generatefs.sh -d $DATADIR/icmr/preprocessed -k $DATADIR/icmr_key.bin -i $DATADIR/icmr.img
./generatefs.sh -d $DATADIR/cowin/preprocessed -k $DATADIR/cowin_key.bin -i $DATADIR/cowin.img
./generatefs.sh -d $DATADIR/index/preprocessed -k $DATADIR/index_key.bin -i $DATADIR/index.img
./generatefs.sh -d $MODELDIR/models -k $MODELDIR/model_key.bin -i $MODELDIR/model.img
rm -rf $MODELDIR/output
mkdir -p $MODELDIR/output
./generatefs.sh -d $MODELDIR/output -k $MODELDIR/output_key.bin -i $MODELDIR/output.img