#ifndef DOC_BM25_FEATURE_HPP
#define DOC_BM25_FEATURE_HPP

#include <cmath>

#include "doc_feature.hpp"

namespace bench {

/**
 * Copied from WANDbl without the template so it is easier to repurpose at
 * runtime.
 */
struct rank_bm25 {
  const double epsilon_score = 1e-6;
  double k1;
  double b;
  size_t num_docs;
  size_t num_terms;
  double avg_doc_len;

  void set_k1(const uint32_t n) { k1 = n / 100.0; }
  void set_b(const uint32_t n) { b = n / 100.0; }
  double calculate_docscore(const double f_qt, const double f_dt,
                            const double f_t, const double W_d) const {
    double w_qt = std::max(
        epsilon_score, std::log((num_docs - f_t + 0.5) / (f_t + 0.5)) * f_qt);
    double K_d = k1 * ((1 - b) + (b * (W_d / avg_doc_len)));
    double w_dt = ((k1 + 1) * f_dt) / (K_d + f_dt);

    return w_dt * w_qt;
  }
};

/**
 * This is a convenience for fetching the term frequency of a term within a
 * document. The purpose of having it this way is to allow for
 * `get_term_frequency` to be easily stubbed out for unit tests.
 */
class doc_data {
public:
  virtual ~doc_data() {}

  virtual size_t get_term_frequency(lemur::api::DOCID_T doc_id,
                                    lemur::api::TERMID_T term_id,
                                    indri::index::Index &index) {
    size_t count = 0;
    auto it = index.docListIterator(term_id);
    it->startIteration();
    if (it->nextEntry(doc_id)) {
      auto doc_data = it->currentEntry();
      count = doc_data->positions.size();
    }

    delete it;
    return count;
  }
};

/**
 * Common to features using BM25.
 */
class doc_bm25_feature : public doc_feature {
protected:
  rank_bm25 ranker;

public:
  doc_data *doc_data_helper;

  doc_bm25_feature(indri_index &idx) : doc_feature(idx) {
    // Assign default helper
    doc_data *helper = new doc_data();
    doc_data_helper = helper;

    ranker.num_docs = _num_docs;
    ranker.num_terms = _coll_len;
    ranker.avg_doc_len = _avg_doc_len;
  }

  ~doc_bm25_feature() { delete doc_data_helper; }

  void bm25_compute(fat_cache_entry &doc, std::vector<std::string> &query_stems,
                    std::string field_str = "") {
    // within query frequency
    std::map<uint64_t, uint32_t> q_ft;
    // within document frequency
    std::map<uint64_t, uint32_t> d_ft;
    // within field frequency
    std::map<uint64_t, uint32_t> f_ft;

    _score_reset();

    for (auto &s : query_stems) {
      auto tid = index.term(s);

      // get within document term frequency
      d_ft[tid] = doc_data_helper->get_term_frequency(doc.id, tid, index);

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

    for (auto &q : q_ft) {
      // skip non-existent terms
      if (q.first == 0) {
        continue;
      }

      if (0 == d_ft.at(q.first)) {
        continue;
      }

      if (!field_str.size()) {
        _score_doc += ranker.calculate_docscore(
            q.second, d_ft.at(q.first),
            index.documentCount(index.term(q.first)), doc.length);
      }

      // Score document fields
      auto fields = term_list->fields();
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

      double field_score = ranker.calculate_docscore(
          q.second, f_ft.at(q.first),
          index.fieldDocumentCount(field_str, index.term(q.first)), field_len);
      _accumulate_score(field_str, field_score);
    }
  }

  void set_helper(doc_data *new_helper) {
    delete doc_data_helper;
    doc_data_helper = std::move(new_helper);
  }
};

} /* bench */

#endif
