#!/bin/bash

set -u

trap 'echo Abort; exit' INT TERM

BASE=$(dirname $0)
ROOT=$BASE/../..
DEST=${1:-$BASE}

mkdir -p $DEST/{runs,models}

learning_rate=0.05
subsample=0.8
tag="GBRT.learning_rate=$learning_rate,subsample=$subsample"

python $ROOT/python/GBDT.py train_GBRT \
    $BASE/data/set1.{train,valid,test}.npz \
    --model_prefix $DEST/models/model.set1.$tag \
    --learning_rate $learning_rate \
    --subsample $subsample \
    --trees '[10,50,100,500,1000]' \
    --nodes '[32,64]' \

python $ROOT/python/GBDT.py predict_GBRT \
    $BASE/data/set1.test.npz \
    $DEST/models/model.set1.$tag \
    --output_trec_run $DEST/runs/run.set1.$tag
