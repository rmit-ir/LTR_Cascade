#pragma once

#include "prob.hpp"

class doc_prob_feature : public doc_feature {

   public:
    doc_prob_feature(Lexicon &lex) : doc_feature(lex) {}

    void compute(query_train &qry, doc_entry &doc, FreqsEntry &freqs, FieldIdMap &field_id_map) {
        for (auto &q : qry.q_ft) {
            // skip non-existent terms
            if (q.first == 0) {
                continue;
            }

            if (freqs.d_ft.find(q.first) == freqs.d_ft.end()) {
                continue;
            }

            _score_doc += calculate_prob(freqs.d_ft.at(q.first), freqs.doc_length);

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

                double field_score = calculate_prob(
                    freqs.f_ft.at(std::make_pair(field_id, q.first)), freqs.field_len[field_id]);
                _accumulate_score(field_str, field_score);
            }
        }

        doc.prob         = _score_doc;
        doc.prob_body    = _score_body;
        doc.prob_title   = _score_title;
        doc.prob_heading = _score_heading;
        doc.prob_inlink  = _score_inlink;
        doc.prob_a       = _score_a;
    }
};
