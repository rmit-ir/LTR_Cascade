#pragma once

#include "cereal/types/vector.hpp"

struct Posting {
    uint32_t docid = 0;
    uint32_t freq = 0;

    template <class Archive>
    void serialize(Archive &archive) {
        archive(docid, freq);
    }
};

// struct PostingList {
//     uint32_t totalCount = 0;
//     std::vector<Posting> list;

//     template <class Archive>
//     void serialize(Archive &archive) {
//         archive(totalCount, list);
//     }
// };

using PostingList = std::vector<Posting>;
using InvertedIndex = std::vector<PostingList>;