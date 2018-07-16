#pragma once

#include <cstdint>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

#include "indri/Index.hpp"

using indri_index = indri::index::Index;

namespace {

static const std::string              _field_title = "title";
static const std::vector<std::string> _fields = {"body", _field_title, "heading", "inlink", "a"};

std::map<uint64_t, uint32_t> calculate_q_freqs(indri_index &             index,
                                               std::vector<std::string> &query_stems) {

    std::map<uint64_t, uint32_t> q_ft;

    for (auto &s : query_stems) {
        auto tid = index.term(s);

        auto it = q_ft.find(tid);
        if (it == q_ft.end()) {
            q_ft[tid] = 1;
        } else {
            ++it->second;
        }
    }
    return q_ft;
}
} // namespace

/**
 * Score segments of a document with a given query.
 */
class doc_feature {
   public:

    indri_index &index;
    uint64_t     _coll_len    = 0;
    uint64_t     _num_docs    = 0;
    double       _avg_doc_len = 0.0;

    double _score_doc     = 0.0;
    double _score_body    = 0.0;
    double _score_title   = 0.0;
    double _score_heading = 0.0;
    double _score_inlink  = 0.0;
    double _score_a       = 0.0;
    // FIXME: implement url score
    double _score_url = 0.0;

    doc_feature(indri_index &idx) : index(idx) {
        _coll_len    = index.termCount();
        _num_docs    = index.documentCount();
        _avg_doc_len = (double)_coll_len / _num_docs;
    }

    virtual ~doc_feature() {}

    inline void _score_reset() {
        _score_doc     = 0.0;
        _score_body    = 0.0;
        _score_title   = 0.0;
        _score_heading = 0.0;
        _score_inlink  = 0.0;
        _score_a       = 0.0;
        _score_url     = 0.0;
    }

    void _accumulate_score(std::string key, double val) {
        if (0 == key.compare(_fields[0])) {
            _score_body += val;
        } else if (0 == key.compare(_fields[1])) {
            _score_title += val;
        } else if (0 == key.compare(_fields[2])) {
            _score_heading += val;
        } else if (0 == key.compare(_fields[3])) {
            _score_inlink += val;
        } else if (0 == key.compare(_fields[4])) {
            _score_a += val;
        } else {
            std::ostringstream oss;
            oss << "unkown field " << key;
            throw std::invalid_argument(oss.str());
        }
    }
};
