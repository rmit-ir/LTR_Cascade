#pragma once

#include "cereal/types/vector.hpp"
#include "cereal/types/string.hpp"

struct Posting {
    uint32_t docid = 0;
    uint32_t freq = 0;

    Posting() = default;
    Posting(uint32_t d, uint32_t f) : docid(d), freq(f) {}

    template <class Archive>
    void serialize(Archive &archive) {
        archive(docid, freq);
    }
};

struct PostingList {
    std::string term;
    uint32_t totalCount = 0;
    std::vector<Posting> list;

    PostingList() = default;
    PostingList(const std::string &t, uint32_t tc) : term(t), totalCount(tc) {}

    template <class Archive>
    void serialize(Archive &archive) {
        archive(term, totalCount, list);
    }
};

using InvertedIndex = std::vector<PostingList>;