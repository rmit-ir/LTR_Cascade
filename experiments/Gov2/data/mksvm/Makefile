# config.sh
#
CORPUS_PATH := /Users/amallia/Development/GOV2

# common.sh
#
BIN := ../../../../external/local/bin
CONFIGDIR := ../config

.PHONY: all links indri prior unigram bigram wand termfeat docfeat clean

.DELETE_ON_ERROR:

all: all.svm

links: gov2_links/.done
indri: gov2_indri/manifest
prior: gov2_indri/prior/pagerank
unigram: gov2_unigram.txt
bigram: gov2_bigram.txt
wand: gov2_wand/.done
termfeat: gov2_termfeat.csv
docfeat: gov2_docfeat.csv


gov2_links/.done:
	$(BIN)/harvestlinks -corpus=$(CORPUS_PATH) -output=gov2_links
	touch $@

gov2_indri/manifest: gov2_links/.done
	rm -rf gov2_indri
	$(BIN)/IndriBuildIndex $(CONFIGDIR)/gov2.param

gov2_indri/prior/pagerank: gov2_indri/manifest pagerank.prior
	$(BIN)/makeprior -index=gov2_indri -input=pagerank.prior -name=pagerank

pagerank.prior: pagerank.prior.xz
	xz -fdk $<

# Perform a Stage0 run and replace 'Q0' with relevance labels
gov2_wand/.done: gov2_indri/manifest
	$(BIN)/mk_wand_idx gov2_indri gov2_wand
	touch $@

stage0.run: gov2_wand/.done gov2-all-kstem.qry
	$(BIN)/wand_search -c gov2_wand -q gov2-all-kstem.qry -k 10000 -i
	rm wand-time.log
	./label.awk gov2.qrels wand-trec.run > $@

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

gov2_unigram.txt: gov2.fwd gov2.inv
	$(BIN)/generate_term_features -i gov2.inv -d gov2.lens  -o $@

gov2_bigram.txt: gov2_bigram.inv
	$(BIN)/generate_term_features -i $< -d gov2.lens -o $@

gov2_docfeat.csv: gov2_indri/manifest stage0.run gov2.fwd gov2.lex
	$(BIN)/fgtrain gov2-all-kstem.qry stage0.run gov2_indri gov2.fwd gov2.lex $@

gov2_termfeat.csv: gov2_indri/manifest gov2_unigram.txt gov2_bigram.txt gov2.lex
	$(BIN)/preret_csv gov2-all-kstem.qry gov2_unigram.txt gov2_bigram.txt gov2.lex\
	    | sed -E 's/\-?nan/0.00000/g' > $@

termfeat.tmp: gov2_docfeat.csv gov2_termfeat.csv
	./termfeat_expand.awk gov2_docfeat.csv gov2_termfeat.csv | sed -E 's/[^,]+,//' > $@

all.csv: gov2_docfeat.csv termfeat.tmp
	paste -d, gov2_docfeat.csv termfeat.tmp > $@

all.svm: all.csv
	./csv2svm.awk $< > $@
# Clean up
clean:
	rm -rf gov2_indri gov2_links gov2_wand pagerank.ivl *.csv *.svm
