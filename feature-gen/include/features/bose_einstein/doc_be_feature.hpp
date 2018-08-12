#pragma once
#include "be.hpp"
class doc_be_feature : public doc_feature {

public:
  doc_be_feature(Lexicon &lex) : doc_feature(lex) {}

  void compute(query_train &qry, doc_entry &doc, Document &doc_idx, FieldIdMap &field_id_map) {
    for (auto &q : qry.q_ft) {
      // skip non-existent terms
      if (q.first == 0) {
        continue;
      }

      if (doc_idx.freq(q.first) == 0) {
        continue;
      }

      _score_doc +=
          calculate_be(doc_idx.freq(q.first), lexicon[q.first].term_count(), _num_docs, _avg_doc_len,
                        doc_idx.length());

      // Score document fields
      for (const std::string &field_str : _fields) {
        int field_id = field_id_map[field_str];
        if (field_id < 1) {
          // field is not indexed
          continue;
        }

        if ( doc_idx.field_len(field_id) == 0) {
          continue;
        }
        if (doc_idx.freq(field_id, q.first) == 0) {
          continue;
        }

        int field_term_cnt = lexicon[q.first].field_term_count(field_id);
        if (0 == field_term_cnt) {
          continue;
        }

        double field_score =
            calculate_be(doc_idx.freq(field_id, q.first), field_term_cnt, _num_docs, _avg_doc_len, doc_idx.field_len(field_id));
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
