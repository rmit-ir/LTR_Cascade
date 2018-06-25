#ifndef DOC_PROB_FEATURE_HPP
#define DOC_PROB_FEATURE_HPP

#include "doc_feature.hpp"

/**
 * Probability
 */
class doc_prob_feature : public doc_feature {
  double _calculate_prob(double d_f, double dlen) { return (double)d_f / dlen; }

public:
  doc_prob_feature(indri_index &idx) : doc_feature(idx) {}

  void compute(fat_cache_entry &doc, freqs_entry &freqs) {

    _score_reset();

    const indri::index::TermList *term_list = doc.term_list;
    auto &doc_terms = term_list->terms();

    for (auto &q : freqs.q_ft) {
      // skip non-existent terms
      if (q.first == 0) {
        continue;
      }

      if (0 == freqs.d_ft.at(q.first)) {
        continue;
      }

      _score_doc += _calculate_prob(freqs.d_ft.at(q.first), doc_terms.size());

      // Score document fields
      for (const std::string &field_str : _fields) {
        int field_id = index.field(field_str);
        if (field_id < 1) {
          // field is not indexed
          continue;
        }

        if (0 == freqs.field_len[field_id]) {
          continue;
        }
        if (0 == freqs.f_ft.at(std::make_pair(field_id, q.first))) {
          continue;
        }

        double field_score = _calculate_prob(freqs.f_ft.at(std::make_pair(field_id, q.first)), freqs.field_len[field_id]);
        _accumulate_score(field_str, field_score);
      }
    }

    doc.prob = _score_doc;
    doc.prob_body = _score_body;
    doc.prob_title = _score_title;
    doc.prob_heading = _score_heading;
    doc.prob_inlink = _score_inlink;
    doc.prob_a = _score_a;
  }
};

#endif
