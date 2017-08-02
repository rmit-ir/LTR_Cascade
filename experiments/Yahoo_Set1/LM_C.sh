#!/bin/bash

set -u

trap 'echo Abort; exit' INT TERM

if [ $# -lt 1 ]; then
    echo "usage: $(basename $0) <dest> <n_threads>" >&2
    exit 0
fi

BASE=$(dirname $0)
ROOT=$BASE/../..
DEST=${1?Missing DEST}
N_THREADS=${2?Missing N_THREADS}

mkdir -p $DEST/{runs,models,eval_results}

run_cascade() {
    alphas="$1"
    strategy=cost
    tag="set1.LM_C.alpha=$alphas"

    echo $alphas
    python $ROOT/python/Cascade.py train $strategy \
	$BASE/data/set1.{train,valid,test}.npz $BASE/data/{costs,importance}.txt \
	--n_stages 3 --alpha "$alphas" --epochs 10 --model_prefix "$DEST/models/model.$tag"
    python $ROOT/python/Cascade.py predict \
	$BASE/data/{set1.valid.npz,costs.txt} "$DEST/models/model.$tag" \
	--output_eval "$DEST/eval_results/valid.$tag"
    python $ROOT/python/Cascade.py predict \
	$BASE/data/{set1.test.npz,costs.txt} "$DEST/models/model.$tag" \
	--output_trec_run "$DEST/runs/run.$tag" --output_eval "$DEST/eval_results/test.$tag"
}

export -f run_cascade
export BASE ROOT DEST

which parallel >/dev/null || { echo "error: GNU parallel is not installed" >&2; }

parallel -j$N_THREADS run_cascade {} ::: $(python $BASE/gen_alphas.py -e 500)
