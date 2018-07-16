#pragma once

#include <cstdint>

#include "indri/QueryEnvironment.hpp"

using docid_t           = lemur::api::DOCID_T;
using query_environment = indri::api::QueryEnvironment;

class query_environment_adapter {
    query_environment *env = nullptr;

   public:
    query_environment_adapter() {}

    query_environment_adapter(query_environment *query_env) : env(query_env) {}

    virtual void add_index(const std::string &path) { env->addIndex(path); }

    virtual double expression_count(const std::string &expr) { return env->expressionCount(expr); }

    virtual std::string stem_term(const std::string &term) { return env->stemTerm(term); }

    virtual uint64_t document_count() { return env->documentCount(); }

    virtual uint64_t term_count() { return env->termCount(); }

    virtual std::vector<std::string> document_metadata(const std::vector<docid_t> &doc_ids,
                                                       const std::string &         attribute_name) {
        return env->documentMetadata(doc_ids, attribute_name);
    }

    virtual std::vector<docid_t> document_ids_from_metadata(const std::string &             name,
                                                            const std::vector<std::string> &value) {
        return env->documentIDsFromMetadata(name, value);
    }

    virtual std::vector<std::string> field_list() { return env->fieldList(); }

    virtual std::vector<indri::api::ParsedDocument *> documents(
        const std::vector<docid_t> &doc_ids) {
        return env->documents(doc_ids);
    }
};
