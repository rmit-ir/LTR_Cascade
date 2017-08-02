/**
 * Benchmark document prior access.
 */

#include <algorithm>
#include <cstdlib>
#include <chrono>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>

#include "indri_bridge.hpp"
#include "query_environment_adapter.hpp"
#include "fat.hpp"
#include "trec_run_file.hpp"

struct bench_entry {
  int qid;
  uint64_t total;
  std::vector<uint64_t> execs;
};

static void display_usage(char *name) {
  std::cerr << "usage: " << name << " <trec file> <indri repository>"
            << std::endl;
  exit(EXIT_FAILURE);
}

int main(int argc, char **argv) {
  if (argc < 3) {
    display_usage(argv[0]);
  }

  std::string trec_file = argv[1];
  std::string repo_path = argv[2];

  query_environment indri_env;
  query_environment_adapter qry_env(&indri_env);
  qry_env.add_index(repo_path);

  indri::collection::Repository repo;
  repo.openRead(repo_path);
  auto index = (*repo.indexes())[0];

  fat_cache *idx_cache = new fat_cache(&repo, index, index->documentCount());
  idx_cache->init();

  // load trec run file
  std::ifstream ifs;
  ifs.open(trec_file);
  trec_run_file trec_run(ifs);
  trec_run.parse();
  ifs.close();
  ifs.clear();

  int qry_id = 701;
  std::vector<std::string> docnos = trec_run.get_result(qry_id);
  std::vector<docid_t> docids =
      qry_env.document_ids_from_metadata("docno", docnos);

  std::cerr << "loading documents..." << std::endl;
  idx_cache->add_docs(docids);

  std::cerr << "timing prior access for " << docnos.size() << " documents...";
  using clock = std::chrono::high_resolution_clock;
  bench_entry entry;
  entry.qid = qry_id;
  entry.total = 0;
  double val;

  for (size_t i = 0; i < docids.size(); ++i) {
    auto start = clock::now();
    auto it = idx_cache->get_entry(docids[i]);
    val = it->second.pagerank;
    auto stop = clock::now();
    auto load_time =
        std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);
    entry.execs.push_back(load_time.count());
    entry.total += load_time.count();
    (void)val;
  }
  std::cerr << "done." << std::endl;

  std::sort(entry.execs.begin(), entry.execs.end());
  size_t median = entry.execs.size() / 2;
  std::cout << entry.execs[0] << " " << entry.execs[median] << " "
            << entry.execs[entry.execs.size() - 1] << std::endl;

  delete idx_cache;

  return 0;
}
