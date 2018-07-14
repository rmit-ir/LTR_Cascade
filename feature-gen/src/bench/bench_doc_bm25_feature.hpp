#ifndef BENCH_DOC_BM25_FEATURE_HPP
#define BENCH_DOC_BM25_FEATURE_HPP

#include <cstdint>
#include <iostream>
#include <map>
#include <string>
#include <utility>

#include "feature_interface.hpp"
#include "query_environment_adapter.hpp"
#include "bm25.hpp"
#include "fat.hpp"

#include "indri/Repository.hpp"
#include "indri/CompressedCollection.hpp"

/**
 * BM25 with various parameters on document body and fields.
 *
 * This doesn't use `feature_interface` as it represents a bag of features
 * relating to BM25.
 */
class bench_doc_bm25_feature {
  using indri_index = indri::index::Index;

  indri_index &index;
  my_rank_bm25<90, 40> ranker_atire;
  // Doc body scores
  double score_atire = 0;
  // Various tags scores
  double score_atire_title = 0;
  double score_atire_heading = 0;
  double score_atire_inlink = 0;

public:
  bench_doc_bm25_feature(indri_index &idx) : index(idx) {
    ranker_atire.num_docs = index.documentCount();
    ranker_atire.num_terms = index.termCount();
    ranker_atire.avg_doc_len =
        (double)ranker_atire.num_terms / ranker_atire.num_docs;
  }

  void compute(fat_cache_entry &doc, std::vector<std::string> &query_stems,
               std::string field_str = "") {
    // within query frequency
    std::map<uint64_t, uint32_t> q_ft;
    // within document frequency
    std::map<uint64_t, uint32_t> d_ft;
    // within field frequency
    std::map<uint64_t, uint32_t> f_ft;

    score_atire = 0;
    score_atire_title = 0;
    score_atire_heading = 0;
    score_atire_inlink = 0;

    for (auto &s : query_stems) {
      auto tid = index.term(s);

      // initialise field term frequency
      f_ft[tid] = 0;

      // get document term frequency
      d_ft[tid] = index.documentCount(s);

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
    /* for (auto tid : doc_terms) { */
    /*   auto it = d_ft.find(tid); */
    /*   if (it == d_ft.end()) { */
    /*     continue; */
    /*   } */
    /*   ++it->second; */
    /* } */

    for (auto &q : q_ft) {
      // skip non-existent terms
      if (q.first == 0) {
        continue;
      }

      // Score document BM25
      if (!field_str.length()) {
        score_atire += ranker_atire.calculate_docscore(
            q.second, d_ft.at(q.first), index.termCount(index.term(q.first)),
            doc_terms.size(), true);
        continue;
      }

      // Score document field BM25
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

        field_len = f.end - f.begin;
        for (size_t i = f.begin; i < f.end; ++i) {
          auto it = f_ft.find(doc_terms[i]);
          if (it == f_ft.end()) {
            f_ft[doc_terms[i]] = 0;
          } else {
            ++it->second;
          }
        }
      }

      double field_score = ranker_atire.calculate_docscore(
          q.second, f_ft.at(q.first),
          index.fieldTermCount(field_str, index.term(q.first)), field_len,
          true);
      if (field_str.compare("title") == 0) {
        score_atire_title += field_score;
      } else if (field_str.compare("heading") == 0) {
        score_atire_heading += field_score;
      } else if (field_str.compare("inlink") == 0) {
        score_atire_inlink += field_score;
      }
    }

    doc.bm25_atire = score_atire;
    doc.bm25_atire_title = score_atire_title;
    doc.bm25_atire_heading = score_atire_heading;
    doc.bm25_atire_inlink = score_atire_inlink;
  }
};

#endif
