# Experiment: Yahoo! Set 1 #

The experiment is based on the Yahoo! Learning to Rank Challenge data version 2.0.  

## Data ##

Go to the [Yahoo!  Webscope](https://webscope.sandbox.yahoo.com) website, and
download the _C14B - Yahoo! Learn to Rank Challenge version 2.0_ data from
there.

By default, the data should be fully decompressed and placed under
`data/Webscope_C14B` (including the file `ltrc_yahoo.tgz`).  To build the data,
simply do:

    make -C data

or when alternative data path is used:

    make SRC=/path/to/Webscope_C14B -C data

A number of files under `data` will be used in the experiment, including
training/validation/test data (`set1.*.npz`), qrels (`qrels.set1`), and
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
(replace `model.foo` and the like with actual filenames):

    bash TreeCascade.sh ./TreeCascade ./LM_C/models/model.foo
    bash TreeCascade.sh ./TreeCascade ./LM_E/models/model.bar
    bash TreeCascade.sh ./TreeCascade ./LM_F/models/model.baz

This script will train all 3 tree cascade models using the best tree params found in the 
respective baselines, and store them under `TreeCascade`.
