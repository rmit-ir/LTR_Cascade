#pragma once

#include "doc_lm_dir_feature.hpp"

class doc_lm_dir_1000_feature : public doc_lm_dir_feature<1000> {

   public:
    doc_lm_dir_1000_feature(indri_index &idx) : doc_lm_dir_feature(idx) {}

    void compute(doc_entry &doc, FreqsEntry &freqs) {
        lm_dir_compute(doc, freqs);
        doc.lm_dir_1000         = _score_doc;
        doc.lm_dir_1000_body    = _score_body;
        doc.lm_dir_1000_title   = _score_title;
        doc.lm_dir_1000_heading = _score_heading;
        doc.lm_dir_1000_inlink  = _score_inlink;
        doc.lm_dir_1000_a       = _score_a;
    }
};
