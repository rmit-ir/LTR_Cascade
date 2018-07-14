#pragma once
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

      _score_doc += _calculate_dfr(
          freqs.d_ft.at(q.first), index.termCount(index.term(q.first)),
          index.documentCount(index.term(q.first)), freqs.doc_length);

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

        double field_score = _calculate_dfr(freqs.f_ft.at(std::make_pair(field_id, q.first)), field_term_cnt,
                                            field_doc_cnt, freqs.field_len[field_id]);
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
