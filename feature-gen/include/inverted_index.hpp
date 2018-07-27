#pragma once

#include "cereal/types/vector.hpp"
#include "cereal/types/string.hpp"
#include "cereal/types/map.hpp"

struct PostingList {
    std::string term;
    uint32_t totalCount = 0;
    std::map<uint32_t, uint32_t> list;

    PostingList() = default;
    PostingList(const std::string &t, uint32_t tc) : term(t), totalCount(tc) {}

    uint32_t freq(uint32_t docid) const {
        if (list.find(docid) == list.end()) {
            return 0;
        }
        return list.at(docid);
    }

    template <class Archive>
    void serialize(Archive &archive) {
        archive(term, totalCount, list);
    }
};

using InvertedIndex = std::vector<PostingList>;