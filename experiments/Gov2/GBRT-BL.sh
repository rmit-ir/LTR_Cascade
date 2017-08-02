#!/bin/bash

set -u

trap 'echo Abort; exit' INT TERM

BASE=$(dirname $0)
ROOT=$BASE/../..
DEST=${1:-$(dirname $0)}

mkdir -p $DEST/{runs,models}

model=GBRT
learning_rate=0.05
subsample=0.8
tag="learning_rate=$learning_rate,subsample=$subsample"

for i in 1 2 3 4 5; do
    python $ROOT/python/GBDT.py train_${model} \
	$BASE/data/{train,valid,test}-f${i}.npz \
	--model_prefix $DEST/models/model.gov2.${model}.f${i}.$tag \
	--learning_rate $learning_rate \
	--subsample $subsample \
	--trees '[500]' \
	--nodes '[32]'

    python $ROOT/python/GBDT.py predict_${model} \
	$BASE/data/test-f${i}.npz \
	$DEST/models/model.gov2.${model}.f${i}.$tag \
	--output_trec_run $DEST/runs/run.gov2.${model}.f${i}.$tag
done

echo "Run '$DEST/runs/run.gov2.${model}.all.$tag' generated"
cat $DEST/runs/run.gov2.${model}.f{1,2,3,4,5}.$tag \
    | stdbuf -o0 python $BASE/cut_to_1000.py \
    > $DEST/runs/run.gov2.${model}.all.$tag
