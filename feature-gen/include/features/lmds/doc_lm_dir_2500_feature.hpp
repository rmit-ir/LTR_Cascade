#pragma once

#include "doc_lm_dir_feature.hpp"

class doc_lm_dir_2500_feature : public doc_lm_dir_feature<2500> {

   public:
    doc_lm_dir_2500_feature(indri_index &idx) : doc_lm_dir_feature(idx) {}

    void compute(doc_entry &doc, FreqsEntry &freqs) {
        lm_dir_compute(doc, freqs);
        doc.lm_dir_2500         = _score_doc;
        doc.lm_dir_2500_body    = _score_body;
        doc.lm_dir_2500_title   = _score_title;
        doc.lm_dir_2500_heading = _score_heading;
        doc.lm_dir_2500_inlink  = _score_inlink;
        doc.lm_dir_2500_a       = _score_a;
    }
};
