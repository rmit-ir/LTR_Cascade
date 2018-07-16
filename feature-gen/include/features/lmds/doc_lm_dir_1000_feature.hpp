#pragma once

/**
 * Language model with Dirichlet smoothing.
 */
class doc_lm_dir_1000_feature : public doc_feature {
    const double _mu = 1000.0;

    double _calculate_lm(uint32_t d_f, uint64_t c_f, uint32_t dlen, uint64_t clen, double mu) {
        double numerator   = d_f + mu * c_f / clen;
        double denominator = dlen + mu;
        return (std::log(numerator / denominator));
    }

   public:
    doc_lm_dir_1000_feature(indri_index &idx) : doc_feature(idx) {}

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

            _score_doc += _calculate_lm(freqs.d_ft.at(q.first),
                                        index.termCount(index.term(q.first)),
                                        freqs.doc_length,
                                        _coll_len,
                                        _mu);

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

                double field_score =
                    _calculate_lm(freqs.f_ft.at(std::make_pair(field_id, q.first)),
                                  index.fieldTermCount(field_str, index.term(q.first)),
                                  freqs.field_len[field_id],
                                  _coll_len,
                                  _mu);
                _accumulate_score(field_str, field_score);
            }
        }

        doc.lm_dir_1000         = _score_doc;
        doc.lm_dir_1000_body    = _score_body;
        doc.lm_dir_1000_title   = _score_title;
        doc.lm_dir_1000_heading = _score_heading;
        doc.lm_dir_1000_inlink  = _score_inlink;
        doc.lm_dir_1000_a       = _score_a;
    }
};
