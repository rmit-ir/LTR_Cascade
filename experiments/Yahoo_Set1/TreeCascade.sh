#!/bin/bash

set -u

trap 'echo Abort; exit' INT TERM

if [ $# -lt 2 ]; then
    echo "usage: $(basename $0) <dest> <model>" >&2
    exit 0
fi

BASE=$(dirname $0)
ROOT=$BASE/../..
DEST=${1?Missing DEST}
MODEL=${2?Missing Model}

mkdir -p $DEST/{runs,models,eval_results}

name=$(basename $MODEL)
suffix=${name#model.set1.}

if [[ "$name" != "model.set1.$suffix" ]]; then
    echo "inconsistent model name: $name model.set1.$suffix" >&2
    exit 1
fi

for model_type in LambdaMART GBRT GBDT; do
    tag="set1.${model_type}.$suffix"
    python $ROOT/python/Cascade.py retrain $model_type \
	$BASE/data/set1.{train,valid}.npz $MODEL "$DEST/models/model.$tag" \
	--learning_rate 0.05 --subsample 0.8 --trees '[1000]' --nodes '[64]'
    python $ROOT/python/Cascade.py predict \
	$BASE/data/{set1.test.npz,costs.txt} "$DEST/models/model.$tag" \
	--output_trec_run "$DEST/runs/run.$tag" --output_eval "$DEST/eval_results/test.$tag"
done
