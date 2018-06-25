#ifndef DOC_LM_DIR_FEATURE_HPP
#define DOC_LM_DIR_FEATURE_HPP

#include "doc_feature.hpp"

/**
 * Language model with Dirichlet smoothing, mu set to Indri's default.
 */
class doc_lm_dir_feature : public doc_feature {
  const double _mu = 2500.0;

  double _calculate_lm(uint32_t d_f, uint64_t c_f, uint32_t dlen, uint64_t clen,
                       double mu) {
    double numerator = d_f + mu * c_f / clen;
    double denominator = dlen + mu;
    return (std::log(numerator / denominator));
  }

public:
  doc_lm_dir_feature(indri_index &idx) : doc_feature(idx) {}

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

      _score_doc +=
          _calculate_lm(freqs.d_ft.at(q.first), index.termCount(index.term(q.first)),
                        doc_terms.size(), _coll_len, _mu);

      // Score document fields
      auto fields = term_list->fields();
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

        double field_score =
            _calculate_lm(freqs.f_ft.at(std::make_pair(field_id, q.first)),
                          index.fieldTermCount(field_str, index.term(q.first)),
                          freqs.field_len[field_id], _coll_len, _mu);
        _accumulate_score(field_str, field_score);
      }
    }

    doc.lm_dir_2500 = _score_doc;
    doc.lm_dir_2500_body = _score_body;
    doc.lm_dir_2500_title = _score_title;
    doc.lm_dir_2500_heading = _score_heading;
    doc.lm_dir_2500_inlink = _score_inlink;
    doc.lm_dir_2500_a = _score_a;
  }
};

#endif
