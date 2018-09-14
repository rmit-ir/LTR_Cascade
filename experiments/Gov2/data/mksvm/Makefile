# config.sh
#
# Create a file 'local.mk' and set `CORPUS_PATH` path as per example below:
#
#   CORPUS_PATH := /path/to/gov2/corpus
include local.mk

# common.sh
#
BIN := ../../../../external/local/bin
CONFIGDIR := ../config

.PHONY: all links indri prior unigram bigram termfeat docfeat clean

.DELETE_ON_ERROR:

all: all.svm

links: gov2_links/.done
indri: gov2_indri/manifest
prior: gov2_indri/prior/pagerank
unigram: gov2_unigram.txt
bigram: gov2_bigram.txt
termfeat: gov2_termfeat.csv
docfeat: gov2_docfeat.csv


gov2_links/.done:
	$(BIN)/harvestlinks -corpus=$(CORPUS_PATH) -output=gov2_links
	touch $@

gov2_indri/manifest: gov2_links/.done
	rm -rf gov2_indri
	$(BIN)/IndriBuildIndex $(CONFIGDIR)/gov2.param

pagerank.prior: pagerank.prior.xz
	xz -fdk $<

gov2_indri/prior/pagerank: gov2_indri/manifest pagerank.prior
	$(BIN)/makeprior -index=gov2_indri -input=pagerank.prior -name=pagerank

# Perform a Stage0 run and replace 'Q0' with relevance labels
stage0.run: gov2-bow.qry
	./stage0.sh
	./label.awk gov2.qrels gov2-bow.run > $@
	$(RM) gov2-bow.run

gov2.fwd: gov2_indri/prior/pagerank
	$(BIN)/create_forward_index gov2_indri $@

gov2.inv: gov2_indri/prior/pagerank
	$(BIN)/create_inverted_index gov2_indri $@

gov2.lex: gov2_indri/prior/pagerank
	$(BIN)/create_lexicon gov2_indri $@

gov2.lens: gov2_indri/prior/pagerank
	$(BIN)/create_doc_lens gov2_indri $@

gov2_bigram.inv: gov2-all-kstem.qry gov2.lex
	$(BIN)/create_bigram_inverted_index -r gov2_indri -q gov2-all-kstem.qry -l gov2.lex -o $@

gov2_unigram.txt: gov2.inv gov2.lens
	$(BIN)/generate_term_features -i gov2.inv -d gov2.lens -o $@

gov2_bigram.txt: gov2_bigram.inv gov2.lens
	$(BIN)/generate_term_features -i $< -d gov2.lens -o $@

gov2_docfeat.csv: gov2_indri/manifest stage0.run gov2.fwd gov2.lex
	$(BIN)/generate_document_features gov2-all-kstem.qry stage0.run gov2_indri gov2.fwd gov2.lex $@

gov2_termfeat.csv: gov2_indri/manifest gov2_unigram.txt gov2_bigram.txt gov2.lex
	$(BIN)/preret_csv gov2-all-kstem.qry gov2_unigram.txt gov2_bigram.txt gov2.lex\
	    | sed -E 's/\-?nan/0.00000/g' > $@

termfeat.tmp: gov2_docfeat.csv gov2_termfeat.csv
	./termfeat_expand.awk gov2_docfeat.csv gov2_termfeat.csv | sed -E 's/[^,]+,//' > $@

all.csv: gov2_docfeat.csv termfeat.tmp
	paste -d, gov2_docfeat.csv termfeat.tmp > $@

all.svm: all.csv
	./csv2svm.awk $< > $@

clean:
	$(RM) -r gov2_indri gov2_links pagerank.prior *.csv *.svm \
		gov2.lex gov2.fwd gov2.inv gov2.lens gov2_unigram.txt \
		stage0.run termfeat.tmp gov2_bigram.inv gov2_bigram.txt
