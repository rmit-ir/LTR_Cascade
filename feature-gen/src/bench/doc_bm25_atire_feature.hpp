#ifndef DOC_BM25_ATIRE_FEATURE_HPP
#define DOC_BM25_ATIRE_FEATURE_HPP

#include "doc_bm25_feature.hpp"

namespace bench {

/**
 * BM25 using Atire's parameters.
 */
class doc_bm25_atire_feature : public doc_bm25_feature {
public:
  doc_bm25_atire_feature(indri_index &idx) : doc_bm25_feature(idx) {}

  void compute(fat_cache_entry &doc, std::vector<std::string> &query_stems,
               std::string field_str = "") {
    ranker.set_k1(90);
    ranker.set_b(40);

    bm25_compute(doc, query_stems, field_str);

    doc.bm25_atire = _score_doc;
    doc.bm25_atire_body = _score_body;
    doc.bm25_atire_title = _score_title;
    doc.bm25_atire_heading = _score_heading;
    doc.bm25_atire_inlink = _score_inlink;
    doc.bm25_atire_a = _score_a;
  }
};

#endif

} /* bench */
