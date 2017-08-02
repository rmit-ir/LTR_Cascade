#ifndef DOC_DFR_FEATURE_HPP
#define DOC_DFR_FEATURE_HPP

#include "doc_feature.hpp"

/**
 * DFR: DPH
 */
class doc_dfr_feature : public doc_feature {
  double _calculate_dfr(uint32_t d_f, uint64_t c_f, uint32_t c_idf,
                        uint32_t dlen) {
    double fp1, ne, ir, prime, rsv;

    fp1 = c_f + 1.0;
    ne = _num_docs * (1.0 - std::pow((_num_docs - 1.0) / _num_docs, c_f));
    ir = std::log2(((double)_num_docs + 1.0) / (ne + 0.5));

    prime = d_f * std::log2(1.0 + (double)_avg_doc_len / (double)dlen);
    rsv = prime * ir * (fp1 / ((double)c_idf * (prime + 1.0)));

    return rsv;
  }

public:
  doc_dfr_feature(indri_index &idx) : doc_feature(idx) {}

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

      _score_doc += _calculate_dfr(
          d_ft.at(q.first), index.termCount(index.term(q.first)),
          index.documentCount(index.term(q.first)), doc_terms.size());

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

        int field_term_cnt =
            index.fieldTermCount(field_str, index.term(q.first));
        if (0 == field_term_cnt) {
          continue;
        }
        int field_doc_cnt =
            index.fieldDocumentCount(field_str, index.term(q.first));
        if (0 == field_doc_cnt) {
          continue;
        }

        double field_score = _calculate_dfr(f_ft.at(q.first), field_term_cnt,
                                            field_doc_cnt, field_len);
        _accumulate_score(field_str, field_score);
      }
    }

    doc.dfr = _score_doc;
    doc.dfr_body = _score_body;
    doc.dfr_title = _score_title;
    doc.dfr_heading = _score_heading;
    doc.dfr_inlink = _score_inlink;
    doc.dfr_a = _score_a;
  }
};

#endif
