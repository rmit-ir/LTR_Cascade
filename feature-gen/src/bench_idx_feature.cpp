#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <ratio>
#include <string>

#include "indri_bridge.hpp"
#include "fat.hpp"
#include "query_train_file.hpp"
#include "trec_run_file.hpp"
#include "document_features.hpp"
#include "docmeta_url_feature.hpp"
#include "doc_bm25_feature.hpp"
#include "doc_proximity_feature.hpp"

#include "fgen_term_qry.h"
#include "query_features.h"
#include "fgen_bigram_qry.h"

#include "indri/Repository.hpp"
#include "indri/CompressedCollection.hpp"

struct bench_entry {
  int qid;
  int fid;
  size_t qlen;
  std::vector<uint64_t> execs_unigram;
  std::vector<uint64_t> execs_bigram;
};

static void display_usage(char *name) {
  std::cerr << "usage: " << name
            << " <query file> <unigram file> <bigram file> <repo>" << std::endl;
  exit(EXIT_FAILURE);
}

char *stdstr_to_cstr(const std::string &s) {
  char *cstr = new char[s.size() + 1];
  std::strcpy(cstr, s.c_str());

  return cstr;
}

int main(int argc, char **argv) {
  using clock = std::chrono::high_resolution_clock;

  if (argc < 5) {
    display_usage(argv[0]);
  }

  std::string query_file = argv[1];
  std::string term_file = argv[2];
  std::string bigram_file = argv[3];
  std::string repo_path = argv[4];

  query_environment indri_env;
  query_environment_adapter qry_env(&indri_env);

  // set Indri repository
  qry_env.add_index(repo_path);

  // `getRepositories` we added to Indri
  std::vector<indri::collection::Repository *> repos =
      indri_env.getRepositories();
  indri::collection::Repository *repo = repos.at(0);
  indri::collection::Repository::index_state state = repo->indexes();
  auto index = (*state)[0];

  fat_cache *idx_cache = new fat_cache(repo, index, index->documentCount());
  idx_cache->init();

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
  std::cerr << "loading static unigram features...";
  termhash_t *termmap = NULL;
  termmap = load_termmap(term_file.c_str());
  std::cerr << " done." << std::endl;

  // load bigram features
  std::cerr << "loading static bigram features...";
  bigramhash_t *bigrammap = NULL;
  bigrammap = load_bigrammap(bigram_file.c_str());
  std::cerr << " done." << std::endl;

  const size_t num_execs = 10000;
  std::vector<bench_entry> dat;

  for (auto &qry : qtfile.get_queries()) {
    std::vector<char *> stems;
    std::transform(qry.stems.begin(), qry.stems.end(),
                   std::back_inserter(stems), stdstr_to_cstr);

    std::cerr << qry.id << std::endl;

    bench_entry entry;
    entry.qid = qry.id;
    entry.qlen = stems.size();
    char *ret;

    // unigram
    for (size_t i = 0; i < num_execs; ++i) {
      size_t feature_id = 1;

      auto start = clock::now();
      ret = fgen_term_qry_main(termmap, qry.id, &stems[0], stems.size(),
                               &feature_id);
      if (0 == strlen(ret)) {
        // skip unigrams not in collection
        break;
      }
      auto stop = clock::now();
      auto run_time =
          std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);
      entry.execs_unigram.push_back(run_time.count());
    }

    if (0 == strlen(ret)) {
      // skip unigrams not in collection
      continue;
    }

    // bigram
    for (size_t i = 0; i < num_execs; ++i) {
      size_t feature_id = 1;

      auto start = clock::now();
      ret = fgen_bigram_qry_main(bigrammap, qry.id, &stems[0], stems.size(),
                                 &feature_id);
      if (0 == strlen(ret)) {
        // skip bigrams not in collection
        break;
      }
      auto stop = clock::now();
      auto run_time =
          std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);
      entry.execs_bigram.push_back(run_time.count());
    }
    dat.push_back(entry);

    for (auto cstr : stems) {
      delete[] cstr;
    }
  }

  for (auto entry : dat) {
    std::sort(entry.execs_unigram.begin(), entry.execs_unigram.end());
    std::sort(entry.execs_bigram.begin(), entry.execs_bigram.end());
    uint64_t median_ugram = 0;
    uint64_t median_bgram = 0;
    if (entry.execs_unigram.size()) {
      median_ugram = entry.execs_unigram[entry.execs_unigram.size() / 2];
    }
    if (entry.execs_bigram.size()) {
      median_bgram = entry.execs_bigram[entry.execs_bigram.size() / 2];
    }

    // query id, query length, median nanoseconds, unigram bigram
    std::cout << entry.qid << " " << entry.qlen << " " << median_ugram << " "
              << median_bgram << std::endl;
  }

  destroy_termhash(termmap);
  destroy_bigramhash(bigrammap);

  return 0;
}
