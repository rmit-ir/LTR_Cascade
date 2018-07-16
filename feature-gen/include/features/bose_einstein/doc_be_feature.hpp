#pragma once
class doc_be_feature : public doc_feature {
  // Doc body
  double score_be = 0;
  // Various tags scores
  double score_be_title = 0;
  double score_be_heading = 0;
  double score_be_inlink = 0;

  double _calculate_be(uint32_t d_f, uint64_t c_f, uint32_t dlen) {
    double l, r, prime, rsv;

    l = std::log(1.0 + (double)c_f / _num_docs);
    r = std::log(1.0 + (double)_num_docs / (double)c_f);
    prime = d_f * std::log(1.0 + _avg_doc_len / (double)dlen);
    rsv = (l + prime * r) / (prime + 1.0);

    return rsv;
  }

public:
  doc_be_feature(indri_index &idx) : doc_feature(idx) {}

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

      _score_doc +=
          _calculate_be(freqs.d_ft.at(q.first), index.termCount(index.term(q.first)),
                        freqs.doc_length);

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

        double field_score =
            _calculate_be(freqs.f_ft.at(std::make_pair(field_id, q.first)), field_term_cnt, freqs.field_len[field_id]);
        _accumulate_score(field_str, field_score);
      }
    }

    doc.be = _score_doc;
    doc.be_body = _score_body;
    doc.be_title = _score_title;
    doc.be_heading = _score_heading;
    doc.be_inlink = _score_inlink;
    doc.be_a = _score_a;
  }
};
