#ifndef DOC_LM_DIR_1000_FEATURE_HPP
#define DOC_LM_DIR_1000_FEATURE_HPP

#include "doc_feature.hpp"

namespace bench {

/**
 * Language model with Dirichlet smoothing.
 */
class doc_lm_dir_1000_feature : public doc_feature {
  const double _mu = 1000.0;

  double _calculate_lm(uint32_t d_f, uint64_t c_f, uint32_t dlen, uint64_t clen,
                       double mu) {
    double numerator = d_f + mu * c_f / clen;
    double denominator = dlen + mu;
    return (std::log(numerator / denominator));
  }

public:
  doc_lm_dir_1000_feature(indri_index &idx) : doc_feature(idx) {}

  void compute(fat_cache_entry &doc, std::vector<std::string> &query_stems) {
    // within query frequency
    std::map<uint64_t, uint32_t> q_ft;
    // within document frequency
    std::map<uint64_t, uint32_t> d_ft;
    // within field frequency
    std::map<uint64_t, uint32_t> f_ft;

    _score_reset();

    for (auto &s : query_stems) {
      auto tid = index.term(s);

      // initialise document term frequency
      d_ft[tid] = 0;
      // initialise field term frequency
      f_ft[tid] = 0;

      // get query term frequency
      auto it = q_ft.find(tid);
      if (it == q_ft.end()) {
        q_ft[tid] = 1;
      } else {
        ++it->second;
      }
    }

    const indri::index::TermList *term_list = doc.term_list;
    auto &doc_terms = term_list->terms();
    for (auto tid : doc_terms) {
      auto it = d_ft.find(tid);
      if (it == d_ft.end()) {
        continue;
      }
      ++it->second;
    }

    for (auto &q : q_ft) {
      // skip non-existent terms
      if (q.first == 0) {
        continue;
      }

      if (0 == d_ft.at(q.first)) {
        continue;
      }

      _score_doc +=
          _calculate_lm(d_ft.at(q.first), index.termCount(index.term(q.first)),
                        doc_terms.size(), _coll_len, _mu);

      // Score document fields
      auto fields = term_list->fields();
      for (const std::string &field_str : _fields) {
        int field_id = index.field(field_str);
        size_t field_len = 0;
        if (field_id < 1) {
          // field is not indexed
          continue;
        }
        for (auto &f : fields) {
          if (f.id != static_cast<size_t>(field_id)) {
            continue;
          }

          field_len += f.end - f.begin;
          for (size_t i = f.begin; i < f.end; ++i) {
            auto it = f_ft.find(doc_terms[i]);
            if (it == f_ft.end()) {
              f_ft[doc_terms[i]] = 1;
            } else {
              ++it->second;
            }
          }
        }

        if (0 == field_len) {
          continue;
        }
        if (0 == f_ft.at(q.first)) {
          continue;
        }

        double field_score =
            _calculate_lm(f_ft.at(q.first),
                          index.fieldTermCount(field_str, index.term(q.first)),
                          field_len, _coll_len, _mu);
        _accumulate_score(field_str, field_score);
      }
    }

    doc.lm_dir_1000 = _score_doc;
    doc.lm_dir_1000_body = _score_body;
    doc.lm_dir_1000_title = _score_title;
    doc.lm_dir_1000_heading = _score_heading;
    doc.lm_dir_1000_inlink = _score_inlink;
    doc.lm_dir_1000_a = _score_a;
  }
};

} /* bench */

#endif
