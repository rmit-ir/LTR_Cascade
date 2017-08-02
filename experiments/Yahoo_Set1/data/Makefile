# train/valid/test: original Yahoo! set 1 split

BASE = ../../..
SRC ?= Webscope_C14B


.PHONY: all set1

all: set1 costs.txt


set1: set1.train.npz set1.valid.npz set1.test.npz qrels.set1

set1.train.npz: $(SRC)/set1.train.txt
set1.valid.npz: $(SRC)/set1.valid.txt
set1.test.npz: $(SRC)/set1.test.txt
qrels.set1: set1.test.npz

%.npz:
	python $(BASE)/python/tools.py make_npz $< $@

qrels.%:
	python $(BASE)/python/tools.py make_qrels $< $@

costs.txt: $(SRC)/featurecost.csv
	python make_costs.py $< > $@
