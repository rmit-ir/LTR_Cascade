# GOV2 stratified 5-fold run
# 
# Each "part" consist of a balanced mix of topics from all 3 years' data.
# Folds are created by pulling together individual parts following the LETOR 3
# paper.  All topics are truncated down to a # depth of 5k docs.

BASE = ../../..

SVM_DIR = ./mksvm
SVM_FILES = $(SVM_DIR)/all.svm
COST_FILE = costs.txt

.DELETE_ON_ERROR:

.PHONY: all 

all: train-f1.npz valid-f1.npz test-f1.npz
all: train-f2.npz valid-f2.npz test-f2.npz
all: train-f3.npz valid-f3.npz test-f3.npz
all: train-f4.npz valid-f4.npz test-f4.npz
all: train-f5.npz valid-f5.npz test-f5.npz
all: costs.txt

part1.svm: $(SVM_FILES)
	python select_topics.py --from-file fold1 $+ > $@
part2.svm: $(SVM_FILES)
	python select_topics.py --from-file fold2 $+ > $@
part3.svm: $(SVM_FILES)
	python select_topics.py --from-file fold3 $+ > $@
part4.svm: $(SVM_FILES)
	python select_topics.py --from-file fold4 $+ > $@
part5.svm: $(SVM_FILES)
	python select_topics.py --from-file fold5 $+ > $@

train-f1.npz: part3-5k.npz part4-5k.npz part5-5k.npz
train-f2.npz: part4-5k.npz part5-5k.npz part1-5k.npz
train-f3.npz: part5-5k.npz part1-5k.npz part2-5k.npz
train-f4.npz: part1-5k.npz part2-5k.npz part3-5k.npz
train-f5.npz: part2-5k.npz part3-5k.npz part4-5k.npz

valid-f1.npz: part2-5k.npz
valid-f2.npz: part3-5k.npz
valid-f3.npz: part4-5k.npz
valid-f4.npz: part5-5k.npz
valid-f5.npz: part1-5k.npz

test-f1.npz: part1-5k.npz
test-f2.npz: part2-5k.npz
test-f3.npz: part3-5k.npz
test-f4.npz: part4-5k.npz
test-f5.npz: part5-5k.npz

train-f1.npz train-f2.npz train-f3.npz train-f4.npz train-f5.npz:
	python $(BASE)/python/tools.py merge_npz $+ $@
valid-f1.npz valid-f2.npz valid-f3.npz valid-f4.npz valid-f5.npz:
	python $(BASE)/python/tools.py merge_npz $+ $@
test-f1.npz test-f2.npz test-f3.npz test-f4.npz test-f5.npz:
	cp $< $@

%.npz: %.svm
	python $(BASE)/python/tools.py make_npz $< $@
%-5k.npz: %.svm
	python $(BASE)/python/tools.py make_npz -k5000 $< $@

# feature costs
costs.txt: $(COST_FILE)
	python make_costs.py $< > $@

