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

run_cascade_cv() {
    alphas="$1"
    strategy=efficiency

    echo $alphas
    for i in 1 2 3 4 5; do 
	tag="gov2.LM_E.f${i}.alpha=$alphas"

	python $ROOT/python/Cascade.py train $strategy \
	    $BASE/data/{train,valid,test}-f${i}.npz \
	    $BASE/data/{costs,importance}.txt \
	    --n_stages 3 --cutoffs '[None,1000,100]' --alpha "$alphas" \
	    --epochs 10 --model_prefix "$DEST/models/model.$tag"
	python $ROOT/python/Cascade.py predict \
	    $BASE/data/{valid-f${i}.npz,costs.txt} \
	    "$DEST/models/model.$tag" \
	    --output_eval "$DEST/eval_results/valid.$tag"
	python $ROOT/python/Cascade.py predict \
	    $BASE/data/{test-f${i}.npz,costs.txt} \
	    "$DEST/models/model.$tag" \
	    --output_trec_run "$DEST/runs/run.$tag" \
	    --output_eval "$DEST/eval_results/test.$tag"
    done
    cat $DEST/runs/run.gov2.LM_E.f{1,2,3,4,5}."alpha=$alphas" \
	| python $BASE/cut_to_1000.py \
	> $DEST/runs/run.gov2.LM_E.all."alpha=$alphas"
}

export -f run_cascade_cv
export BASE ROOT DEST

which parallel >/dev/null || { echo "error: GNU parallel is not installed" >&2; }

parallel -j$N_THREADS run_cascade_cv {} ::: $(python $BASE/gen_alphas.py -small -e 500)
