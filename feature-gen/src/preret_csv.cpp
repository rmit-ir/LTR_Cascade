#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>

#include "query_train_file.hpp"
#include "trec_run_file.hpp"

#include "fgen_bigram_qry.h"
#include "fgen_term_qry.h"
#include "query_features.h"

#include "indri/CompressedCollection.hpp"
#include "indri/Repository.hpp"

static void display_usage(char *name) {
    std::cerr << "usage: " << name << " <query file> <unigram file> <bigram file>"
              << " <indri repository>" << std::endl;
    exit(EXIT_FAILURE);
}

char *stdstr_to_cstr(const std::string &s) {
    char *cstr = new char[s.size() + 1];
    std::strcpy(cstr, s.c_str());

    return cstr;
}

int main(int argc, char **argv) {
    if (argc < 5) {
        display_usage(argv[0]);
    }

    std::string query_file   = argv[1];
    std::string unigram_file = argv[2];
    std::string bigram_file  = argv[3];
    std::string repo_path    = argv[4];

    query_environment         indri_env;
    query_environment_adapter qry_env(&indri_env);

    // set Indri repository
    qry_env.add_index(repo_path);

    indri::collection::Repository repo;
    repo.openRead(repo_path);
    auto index = (*repo.indexes())[0];

    // load query file
    std::ifstream ifs(query_file);
    if (!ifs.is_open()) {
        std::cerr << "Could not open file: " << query_file << std::endl;
        exit(EXIT_FAILURE);
    }
    query_train_file qtfile(ifs, qry_env, index);
    qtfile.parse();
    ifs.close();
    ifs.clear();

    // init static features
    query_features_init(qry_env.document_count(), qry_env.term_count());

    // load unigram features
    std::cerr << "loading unigram features...";
    termhash_t *termmap = NULL;
    termmap             = load_termmap(unigram_file.c_str());
    std::cerr << " done." << std::endl;

    // load bigram features
    std::cerr << "loading bigram features...";
    bigramhash_t *bigrammap = NULL;
    bigrammap               = load_bigrammap(bigram_file.c_str());
    std::cerr << " done." << std::endl;

    for (auto &qry : qtfile.get_queries()) {
        std::vector<char *> stems;
        std::transform(
            qry.stems.begin(), qry.stems.end(), std::back_inserter(stems), stdstr_to_cstr);

        char *buf_unigram, *buf_bigram;

        // dump unigram features
        buf_unigram = fgen_term_qry_main(termmap, qry.id, &stems[0], stems.size());
        // dump bigram features
        buf_bigram = fgen_bigram_qry_main(bigrammap, qry.id, &stems[0], stems.size());
        std::string buf_common(buf_unigram);
        buf_common.append(buf_bigram);
        free(buf_unigram);
        free(buf_bigram);

        std::cout << qry.id;
        if (buf_common.length()) {
            std::cout << buf_common;
        }
        std::cout << std::endl;

        for (auto cstr : stems) {
            delete[] cstr;
        }
    }

    destroy_termhash(termmap);
    destroy_bigramhash(bigrammap);

    return 0;
}
