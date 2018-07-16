#pragma once
/**
 * Probability
 */
class doc_prob_feature : public doc_feature {
    double _calculate_prob(double d_f, double dlen) { return (double)d_f / dlen; }

   public:
    doc_prob_feature(indri_index &idx) : doc_feature(idx) {}

    void compute(doc_entry &doc, FreqsEntry &freqs) {

        _score_reset();

        for (auto &q : freqs.q_ft) {
            // skip non-existent terms
            if (q.first == 0) {
                continue;
            }

            if (freqs.d_ft.find(q.first) == freqs.d_ft.end()) {
                continue;
            }

            _score_doc += _calculate_prob(freqs.d_ft.at(q.first), freqs.doc_length);

            // Score document fields
            for (const std::string &field_str : _fields) {
                int field_id = index.field(field_str);
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

                double field_score = _calculate_prob(
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
