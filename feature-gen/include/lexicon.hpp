#pragma once

#include "cereal/types/map.hpp"
#include "cereal/types/string.hpp"
#include "cereal/types/vector.hpp"

#include <map>

struct Counts {
    uint64_t document_count = 0;
    uint64_t term_count     = 0;

    Counts() = default;
    Counts(uint64_t dc, uint64_t tc) : document_count(dc), term_count(tc) {}

    template <class Archive>
    void serialize(Archive &archive) {
        archive(document_count, term_count);
    }
};

class Term {
   public:
    using FieldCounts = std::map<uint64_t, Counts>;

   private:
    std::string term;
    Counts      counts;
    FieldCounts field_counts;

   public:
    Term() = default;

    Term(std::string t, Counts c, FieldCounts fc) : term(t), counts(c), field_counts(fc){};

    inline uint64_t document_count() const { return counts.document_count; }

    inline uint64_t term_count() const { return counts.term_count; }

    inline uint64_t field_document_count(uint64_t field) const {
        auto it = field_counts.find(field);
        if (it == field_counts.end()) {
            return 0;
        }
        return it->second.document_count;
    }

    inline uint64_t field_term_count(uint64_t field) const {
        auto it = field_counts.find(field);
        if (it == field_counts.end()) {
            return 0;
        }
        return it->second.term_count;
    }

    template <class Archive>
    void serialize(Archive &archive) {
        archive(term, counts, field_counts);
    }
};

class Lexicon {
   public:
    inline uint64_t document_count() const { return counts.document_count; }

    inline uint64_t term_count() const { return counts.term_count; }

    inline const Term &operator[](size_t pos) const { return terms[pos]; }
    inline Term &      operator[](size_t pos) { return terms[pos]; }

    void push_back(const Term &t) { terms.push_back(t); }
    void push_back(Term &&t) { terms.push_back(t); }

    Lexicon() = default;
    Lexicon(Counts c) : counts(c) {}

    template <class Archive>
    void serialize(Archive &archive) {
        archive(counts, terms);
    }

   private:
    Counts            counts;
    std::vector<Term> terms;
};
