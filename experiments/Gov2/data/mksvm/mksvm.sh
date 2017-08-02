#!/usr/bin/env bash

set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $DIR/common.sh
source $CONFIGDIR/config.sh

# Harvest links
$BINDIR/harvestlinks -corpus=$CORPUS_PATH -output=$LINKS_PATH

# Build an Indri index
$BINDIR/IndriBuildIndex $CONFIGDIR/gov2.param

# Extract PageRank priors
xz -fdk $PAGERANK_PRIOR.xz

# Insert pagerank prior
$BINDIR/makeprior -index=$REPO_PATH -input=$PAGERANK_PRIOR -name=pagerank

# Dump the Indri index
$BINDIR/dump_indri $REPO_PATH $DUMP_PATH

# Unigram features
C_LEN=$(awk '{print $2}' $DUMP_PATH/global.txt)
$BINDIR/fgen_term -i $DUMP_PATH/text.inv \
    -d $DUMP_PATH/doc_lens.txt \
    -c $C_LEN > $UNIGRAM_FILE

# Dump bigram unordered window of 8 posting list for `fgen_bigram`
B_SUFFIX=_tmp
awk -F\; '{gsub(" ", ","); print $2}' $QRY > qry$B_SUFFIX
$BINDIR/dump_bigram -i $REPO_PATH -q qry$B_SUFFIX -w8 -u -s $B_SUFFIX
mv u8$B_SUFFIX.txt $BIGRAM_POSTINGS
rm qry$B_SUFFIX

# Bigram features
C_LEN=$(awk '{print $2}' $DUMP_PATH/global.txt)
$BINDIR/fgen_bigram -i $BIGRAM_POSTINGS \
    -d $DUMP_PATH/doc_lens.txt \
    -c $C_LEN > $BIGRAM_FILE
sed -i 's/-nan/0.00000/g' $BIGRAM_FILE

# Build WAND index
$BINDIR/mk_wand_idx $REPO_PATH $WAND_PATH

# Perform a Stage0 run
$BINDIR/wand_search -c $WAND_PATH -q $QRY -k 10000 -i
mv wand-trec.run $STAGE0_RUN
rm wand-time.log

# Replace 'Q0' with relevance labels
./label.awk gov2.qrels $STAGE0_RUN > tmp.run
mv tmp.run $STAGE0_RUN

# Unigram and bigram features
$BINDIR/preret_csv $QRY $UNIGRAM_FILE $BIGRAM_FILE $REPO_PATH > $TERMFEAT_FILE
sed -E -i 's/\-?nan/0.00000/g' $TERMFEAT_FILE

# Document features
SPLIT_DIR=$DIR/split
LINES=372600 # split into about 4 procs
mkdir -p $SPLIT_DIR
cd $SPLIT_DIR && split -dl $LINES $STAGE0_RUN && cd -
for f in $SPLIT_DIR/x??; do
    $BINDIR/fgtrain $QRY \
        $f \
        $REPO_PATH > $f.csv &
done
wait
cat $SPLIT_DIR/x??.csv > $DOCFEAT_FILE

# Combine into SVM light format
./termfeat_expand.awk $DOCFEAT_FILE $TERMFEAT_FILE > termfeat.tmp
sed -E -i 's/[^,]+,//' termfeat.tmp
paste -d, $DOCFEAT_FILE termfeat.tmp > all.csv
./csv2svm.awk all.csv > all.svm
