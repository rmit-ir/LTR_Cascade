#pragma once
#include "lexicon.hpp"
#include "dph.hpp"

class doc_dph_feature : public doc_feature {
   public:
    doc_dph_feature(Lexicon &lex) : doc_feature(lex) {}

    void compute(query_train &qry, doc_entry &doc, FreqsEntry &freqs, FieldIdMap &field_id_map) {
        for (auto &q : qry.q_ft) {
            // skip non-existent terms
            if (q.first == 0) {
                continue;
            }

            if (freqs.d_ft.find(q.first) == freqs.d_ft.end()) {
                continue;
            }

            _score_doc += calculate_dph(
                freqs.d_ft.at(q.first), lexicon[q.first].term_count(), _num_docs, _avg_doc_len, freqs.doc_length);

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

                int field_term_cnt = lexicon[q.first].field_term_count(field_id);
                if (0 == field_term_cnt) {
                    continue;
                }

                double field_score =
                    calculate_dph(freqs.f_ft.at(std::make_pair(field_id, q.first)),
                                   field_term_cnt, _num_docs, _avg_doc_len,
                                   freqs.field_len[field_id]);
                _accumulate_score(field_str, field_score);
            }
        }

        doc.dph         = _score_doc;
        doc.dph_body    = _score_body;
        doc.dph_title   = _score_title;
        doc.dph_heading = _score_heading;
        doc.dph_inlink  = _score_inlink;
        doc.dph_a       = _score_a;
    }
};
