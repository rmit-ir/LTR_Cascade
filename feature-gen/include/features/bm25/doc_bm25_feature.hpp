#pragma once

#include <cmath>

#include "forward_index.hpp"
#include "lexicon.hpp"
#include "field_id.hpp"
#include "bm25.hpp"

class doc_bm25_feature : public doc_feature {
   protected:
    rank_bm25 ranker;

   public:
    doc_bm25_feature(Lexicon &lex) : doc_feature(lex) {

        ranker.num_docs    = _num_docs;
        ranker.avg_doc_len = _avg_doc_len;
    }

    void bm25_compute(query_train &qry, doc_entry &doc, FreqsEntry &freqs, FieldIdMap &field_id_map) {
        for (auto &q : qry.q_ft) {

            // skip non-existent terms
            if (q.first == 0) {
                continue;
            }

            if (freqs.d_ft.find(q.first) == freqs.d_ft.end()) {
                continue;
            }

            _score_doc += ranker.calculate_docscore(q.second,
                                                    freqs.d_ft.at(q.first),
                                                    lexicon[q.first].document_count(),
                                                    doc.length);

            // Score document fields
            for (const std::string &field_str : _fields) {
                int field_id = field_id_map[field_str];
                if (field_id < 1) {
                    // field is not indexed
                    continue;
                }

                if (0 == freqs.field_len[field_id]) {
                    continue;
                }
                if (freqs.f_ft.find(std::make_pair(field_id, q.first)) == freqs.f_ft.end()) {
                    continue;
                }

                double field_score = ranker.calculate_docscore(
                    q.second,
                    freqs.f_ft.at(std::make_pair(field_id, q.first)),
                    lexicon[q.first].field_document_count(field_id),
                    freqs.field_len[field_id]);
                _accumulate_score(field_str, field_score);
            }
        }
    }
};