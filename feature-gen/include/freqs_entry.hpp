#pragma once

#include "cereal/types/map.hpp"
#include "cereal/types/string.hpp"
#include "cereal/types/utility.hpp"
#include "cereal/types/vector.hpp"

struct FreqsEntry {

    // within query frequency
    std::map<uint64_t, uint32_t> q_ft;

    struct UrlStats {
        size_t url_slash_count = 0;
        size_t url_length      = 0;

        template <class Archive>
        void serialize(Archive &archive) {
            archive(url_slash_count, url_length);
        }

    } url_stats;

    struct FieldsStats {
        // The number of times the 'field' tag appears in the document
        std::map<std::string, size_t> tags_count;

        template <class Archive>
        void serialize(Archive &archive) {
            archive(tags_count);
        }
    } fields_stats;

    std::vector<uint64_t> term_list;

    std::map<uint64_t, std::vector<uint64_t>> positions;

    // within document frequency
    std::map<uint64_t, uint32_t> d_ft;

    size_t doc_length = 0;

    double pagerank = 0;

    std::map<std::pair<size_t, uint64_t>, uint32_t> f_ft;

    std::map<size_t, size_t> field_len;
    std::map<size_t, size_t> field_min_len;
    std::map<size_t, size_t> field_max_len;

    std::map<std::string, size_t> field_len_sum_sqrs;

    template <class Archive>
    void serialize(Archive &archive) {
        archive(url_stats,
                fields_stats,
                term_list,
                positions,
                d_ft,
                doc_length,
                pagerank,
                f_ft,
                field_len,
                field_min_len,
                field_max_len,
                field_len_sum_sqrs);
    }
};

using ForwardIndex = std::vector<FreqsEntry>;