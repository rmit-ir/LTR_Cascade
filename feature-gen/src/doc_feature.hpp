#ifndef DOC_FEATURE_HPP
#define DOC_FEATURE_HPP

#include <cstdint>
#include <iostream>
#include <map>
#include <string>
#include <utility>
#include <stdexcept>
#include <sstream>

#include "fat.hpp"

#include "indri/Index.hpp"

using indri_index = indri::index::Index;

namespace {

  static const std::string _field_title = "title";
  static const std::vector<std::string> _fields = {"body", _field_title, "heading",
                                          "inlink", "a"};

  struct freqs_entry {
      // within query frequency
      std::map<uint64_t, uint32_t> q_ft;
      // within document frequency
      std::map<uint64_t, uint32_t> d_ft;
      // within field frequency
      std::map<std::pair<size_t, uint64_t>, uint32_t> f_ft;

      std::map<size_t, size_t> field_len;
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

  freqs_entry calculate_freqs(indri_index &index, fat_cache_entry &doc_entry, std::vector<std::string> &query_stems) {
    doc_data *doc_data_helper = new doc_data();
    freqs_entry freqs;

    for (auto &s : query_stems) {
      auto tid = index.term(s);

      // SLOW
      // get within document term frequency
      freqs.d_ft[tid] =
          doc_data_helper->get_term_frequency(doc_entry.id, tid, index);

      // initialise field term frequency
      for (const std::string &field_str : _fields) {
        int field_id = index.field(field_str);
        freqs.f_ft[std::make_pair(field_id, tid)] = 0;
      }   
      // get query term frequency
      auto it = freqs.q_ft.find(tid);
      if (it == freqs.q_ft.end()) {
        freqs.q_ft[tid] = 1;
      } else {
        ++it->second;
      }
    }
    const indri::index::TermList *term_list = doc_entry.term_list;
    auto &doc_terms = term_list->terms();

    for (auto &q : freqs.q_ft) {
      // skip non-existent terms
      if (q.first == 0) {
        continue;
      }

      if (0 == freqs.d_ft.at(q.first)) {
        continue;
      }

      // Score document fields
      auto fields = term_list->fields();
      for (const std::string &field_str : _fields) {
        int field_id = index.field(field_str);
        freqs.field_len[field_id] = 0;
        if (field_id < 1) {
          // field is not indexed
          continue;
        }
        for (auto &f : fields) {
          if (f.id != static_cast<size_t>(field_id)) {
            continue;
          }

          freqs.field_len[field_id] += f.end - f.begin;

          // SUPER SLOW
          for (size_t i = f.begin; i < f.end; ++i) {
            auto it = freqs.f_ft.find(std::make_pair(field_id, doc_terms[i]));
            if (it == freqs.f_ft.end()) {
              freqs.f_ft[std::make_pair(field_id, doc_terms[i])] = 1;
            } else {
              ++it->second;
            }
          }
        }
      }
    }

    delete doc_data_helper;
    return freqs;
  }
}
/**
 * Score segments of a document with a given query.
 */
class doc_feature {
public:

  const std::string _field_title = "title";
  const std::vector<std::string> _fields = {"body", _field_title, "heading",
                                            "inlink", "a"};

  indri_index &index;
  uint64_t _coll_len = 0;
  uint64_t _num_docs = 0;
  double _avg_doc_len = 0.0;

  double _score_doc = 0.0;
  double _score_body = 0.0;
  double _score_title = 0.0;
  double _score_heading = 0.0;
  double _score_inlink = 0.0;
  double _score_a = 0.0;
  // FIXME: implement url score
  double _score_url = 0.0;

  doc_feature(indri_index &idx) : index(idx) {
    _coll_len = index.termCount();
    _num_docs = index.documentCount();
    _avg_doc_len = (double)_coll_len / _num_docs;
  }

  virtual ~doc_feature() {}

  inline void _score_reset() {
    _score_doc = 0.0;
    _score_body = 0.0;
    _score_title = 0.0;
    _score_heading = 0.0;
    _score_inlink = 0.0;
    _score_a = 0.0;
    _score_url = 0.0;
  }

  void _accumulate_score(std::string key, double val) {
    if (0 == key.compare(_fields[0])) {
      _score_body += val;
    } else if (0 == key.compare(_fields[1])) {
      _score_title += val;
    } else if (0 == key.compare(_fields[2])) {
      _score_heading += val;
    } else if (0 == key.compare(_fields[3])) {
      _score_inlink += val;
    } else if (0 == key.compare(_fields[4])) {
      _score_a += val;
    } else {
      ostringstream oss;
      oss << "unkown field " << key;
      throw invalid_argument(oss.str());
    }
  }
};

#endif
