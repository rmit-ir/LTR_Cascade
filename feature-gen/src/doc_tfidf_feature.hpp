#ifndef DOC_TFIDF_FEATURE_HPP
#define DOC_TFIDF_FEATURE_HPP

#include "doc_feature.hpp"

/**
 * tfidf
 */
class doc_tfidf_feature : public doc_feature {
  double _calculate_tfidf(double d_f, double t_idf, double dlen) {
    double doc_norm = 1.0 / dlen;
    double w_dq = 1.0 + std::log(d_f);
    double w_Qq = std::log(1.0 + ((double)_num_docs / t_idf));

    return (doc_norm * w_dq * w_Qq);
  }

public:
  doc_tfidf_feature(indri_index &idx) : doc_feature(idx) {}

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

      _score_doc += _calculate_tfidf(d_ft.at(q.first),
                                     index.termCount(index.term(q.first)),
                                     doc_terms.size());

      // Score document title, heading, inlink fields
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

        int field_term_cnt =
            index.fieldTermCount(field_str, index.term(q.first));
        if (0 == field_term_cnt) {
          continue;
        }

        double field_score =
            _calculate_tfidf(f_ft.at(q.first), field_term_cnt, field_len);
        _accumulate_score(field_str, field_score);
      }
    }

    doc.tfidf = _score_doc;
    doc.tfidf_body = _score_body;
    doc.tfidf_title = _score_title;
    doc.tfidf_heading = _score_heading;
    doc.tfidf_inlink = _score_inlink;
    doc.tfidf_a = _score_a;
  }
};

#endif
