# Experiment: GOV2 #

This experiment is based on the TREC GOV2 dataset.  To obtain the dataset, go
to the [TREC Web Research
Collections](http://ir.dcs.gla.ac.uk/test_collections/) webpage hosted by
University of Glasgow.

## Data ##

Make a local copy of the configuration files used by Indri and various shell
scripts:

    cp data/config/config.sh.dist data/config/config.sh
    cp data/config/gov2.param.dist data/config/gov2.param


Edit the local configuration files to set the path to your copy of the GOV 2
corpus.  Be sure to make the following changes:

*  In `config.sh`, point `CORPUS_PATH` variable to the GOV2 corpus.

*  In `gov2.param` and `pagerank.param`, replace `/BASEDIR` with the actual
   path to this repo and `/GOV2_CORPUS` to the GOV2 corpus.  Note that in these
   param files the paths have to be *absolute* (Indri likes that better).


Build the feature file:

    make -C data/mksvm

The above script will take a long time as it performs the following tasks:

* Harvest links from the GOV2 corpus
* Build an Indri index
* Run the PageRank algorithm over the corpus
* Insert the PageRank scores as a prior in the Indri index
* Dump the Indri index to ASCII text files
* Compute unigram scores and statistics
* Create bigram posting lists for the queries used in the experiments
* Compute bigram scores and statistics
* Build a WAND index
* Perform a Stage0 run using WAND
* Compute unigram and bigram features
* Compute document features
* Combine and convert to SVM Light format

The final output will be in a full-blown feature file at `data/mksvm/all.svm`
in SVMLight format, covering TREC topics 701-850.

Then, to convert the dataset into NPZ format and 5-fold cross-validated segments that
are used in the experiments:

    make -C data

or when alternative data path is used:

    make SRC=/path/to/data -C data

A number of files under `data` will be used, including training/validation/test
data for individual folds (`train-*.npz`, `valid-*.npz`, and `test-*.npz`) and
costs/importance files (`costs.txt` and `importance.txt`).


## Models ##

### Baselines ###
To run the baseline tree models:

    bash GBDT-BL.sh
    bash GBRT-BL.sh
    bash LambdaMART-BL.sh

### Linear Cascade Models ###
The proposed linear cascade models use randomized search to obtain a good cascade
configuration.  [GNU Parallel](https://www.gnu.org/software/parallel/) is used throughout
this experiments to speed up the search.  To run the linear cascades using 12 parallel processes:

    bash LM_C.sh ./LM_C 12
    bash LM_E.sh ./LM_E 12
    bash LM_F.sh ./LM_F 12

Results and models will be generated in separate directories (e.g., `LM_C`) to prevent clutter.

### Tree Cascade Models ###
To train tree-based cascades based on the feature allocated in some linear cascade model
(replace `model.foo.bar` with the actual filename):

    bash TreeCascade.sh ./Tree_F ./LM_F/models/model.foo.bar

This script will train all 3 tree cascade models using the best tree params found in the 
respective baselines.
