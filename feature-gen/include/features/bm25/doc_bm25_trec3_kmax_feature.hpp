#pragma once
#include "doc_bm25_feature.hpp"

/**
 * BM25 using parameters suggested by Robertson, et al. in TREC 3.
 */
class doc_bm25_trec3_kmax_feature : public doc_bm25_feature {
public:
  doc_bm25_trec3_kmax_feature(indri_index &idx) : doc_bm25_feature(idx) {}

  void compute(doc_entry &doc, FreqsEntry &freqs) {
    ranker.set_k1(200);
    ranker.set_b(75);

    bm25_compute(doc, freqs);

    doc.bm25_trec3_kmax = _score_doc;
    doc.bm25_trec3_kmax_body = _score_body;
    doc.bm25_trec3_kmax_title = _score_title;
    doc.bm25_trec3_kmax_heading = _score_heading;
    doc.bm25_trec3_kmax_inlink = _score_inlink;
    doc.bm25_trec3_kmax_a = _score_a;
  }
};

