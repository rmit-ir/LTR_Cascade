#pragma once

#include <cmath>
#include <freqs_entry.hpp>

/**
 * Copied from WANDbl without the template so it is easier to repurpose at
 * runtime.
 */
struct rank_bm25 {
    const double epsilon_score = 1e-6;
    double       k1;
    double       b;
    size_t       num_docs;
    size_t       num_terms;
    double       avg_doc_len;

    void   set_k1(const uint32_t n) { k1 = n / 100.0; }
    void   set_b(const uint32_t n) { b = n / 100.0; }
    double calculate_docscore(const double f_qt,
                              const double f_dt,
                              const double f_t,
                              const double W_d) const {
        double w_qt =
            std::max(epsilon_score, std::log((num_docs - f_t + 0.5) / (f_t + 0.5)) * f_qt);
        double K_d  = k1 * ((1 - b) + (b * (W_d / avg_doc_len)));
        double w_dt = ((k1 + 1) * f_dt) / (K_d + f_dt);

        return w_dt * w_qt;
    }
};

class doc_bm25_feature : public doc_feature {
   protected:
    rank_bm25 ranker;

   public:
    doc_bm25_feature(indri_index &idx) : doc_feature(idx) {

        ranker.num_docs    = _num_docs;
        ranker.num_terms   = _coll_len;
        ranker.avg_doc_len = _avg_doc_len;
    }

    void bm25_compute(doc_entry &doc, FreqsEntry &freqs) {
        _score_reset();

        for (auto &q : freqs.q_ft) {

            // skip non-existent terms
            if (q.first == 0) {
                continue;
            }

            if (freqs.d_ft.find(q.first) == freqs.d_ft.end()) {
                continue;
            }

            _score_doc += ranker.calculate_docscore(q.second,
                                                    freqs.d_ft.at(q.first),
                                                    index.documentCount(index.term(q.first)),
                                                    doc.length);

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
                double field_score = ranker.calculate_docscore(
                    q.second,
                    freqs.f_ft.at(std::make_pair(field_id, q.first)),
                    index.fieldDocumentCount(field_str, index.term(q.first)),
                    freqs.field_len[field_id]);
                _accumulate_score(field_str, field_score);
            }
        }
    }
};