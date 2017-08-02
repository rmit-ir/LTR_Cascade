#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>

#include "indri_bridge.hpp"
#include "fat.hpp"
#include "query_train_file.hpp"
#include "trec_run_file.hpp"
#include "document_features.hpp"
#include "docmeta_url_feature.hpp"
#include "doc_bm25_atire_feature.hpp"
#include "doc_bm25_trec3_feature.hpp"
#include "doc_bm25_trec3_kmax_feature.hpp"
#include "doc_lm_dir_feature.hpp"
#include "doc_lm_dir_1500_feature.hpp"
#include "doc_lm_dir_1000_feature.hpp"
#include "doc_tfidf_feature.hpp"
#include "doc_prob_feature.hpp"
#include "doc_be_feature.hpp"
#include "doc_dph_feature.hpp"
#include "doc_dfr_feature.hpp"
#include "doc_stream_feature.hpp"
#include "doc_tpscore_feature.hpp"
#include "doc_proximity_feature.hpp"

static void display_usage(char *name) {
  std::cerr << "usage: " << name
            << " <query file> <trec file> <indri repository>" << std::endl;
  exit(EXIT_FAILURE);
}

/**
 * Generate document features and output to CSV format.
 */
int main(int argc, char **argv) {
  if (argc < 4) {
    display_usage(argv[0]);
  }

  std::cout << std::fixed << std::setprecision(5);

  std::string query_file = argv[1];
  std::string trec_file = argv[2];
  std::string repo_path = argv[3];

  query_environment indri_env;
  query_environment_adapter qry_env(&indri_env);
  qry_env.add_index(repo_path);

  indri::collection::Repository repo;
  repo.openRead(repo_path);
  auto index = (*repo.indexes())[0];

  fat_cache *idx_cache = new fat_cache(&repo, index, index->documentCount());
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

  // load trec run file
  ifs.open(trec_file);
  trec_run_file trec_run(ifs);
  trec_run.parse();
  ifs.close();
  ifs.clear();

  document_features features(*index);
  docmeta_url_feature meta_feature;
  doc_bm25_atire_feature f_bm25_atire(*index);
  doc_bm25_trec3_feature f_bm25_trec3(*index);
  doc_bm25_trec3_kmax_feature f_bm25_trec3_kmax(*index);
  doc_proximity_feature prox_feature(*index);
  doc_lm_dir_feature f_lmds_2500(*index);
  doc_lm_dir_1500_feature f_lmds_1500(*index);
  doc_lm_dir_1000_feature f_lmds_1000(*index);
  doc_tfidf_feature tfidf_feature(*index);
  doc_prob_feature prob_feature(*index);
  doc_be_feature be_feature(*index);
  doc_dph_feature dph_feature(*index);
  doc_dfr_feature dfr_feature(*index);
  doc_stream_feature f_stream(*index);
  doc_tpscore_feature f_tpscore(*index);

  for (auto &qry : qtfile.get_queries()) {
    std::vector<double> stage0_scores = trec_run.get_scores(qry.id);
    std::vector<int> docno_labels = trec_run.get_labels(qry.id);
    std::vector<std::string> docnos = trec_run.get_result(qry.id);
    std::vector<docid_t> docids =
        qry_env.document_ids_from_metadata("docno", docnos);

    idx_cache->add_docs(docids);

    using clock = std::chrono::high_resolution_clock;
    auto start = clock::now();

    for (size_t i = 0; i < docids.size(); ++i) {
      auto const docid = docids[i];
      auto const docno = docnos[i];
      auto const label = docno_labels[i];

      auto it = idx_cache->get_entry(docid);
      auto &doc_entry = it->second;

      if (fat_cache_entry::STATE_NEW == doc_entry.state) {
        // number of slashes in doc url
        auto url = qry_env.document_metadata({docid}, "url");
        meta_feature.set_url(url.at(0));
        meta_feature.compute(doc_entry);

        doc_entry.state = fat_cache_entry::STATE_INERT;
      }

      // set original run score as a feature for training
      doc_entry.stage0_score = stage0_scores[i];

      f_bm25_atire.compute(doc_entry, qry.stems);
      f_bm25_trec3.compute(doc_entry, qry.stems);
      f_bm25_trec3_kmax.compute(doc_entry, qry.stems);
      f_lmds_2500.compute(doc_entry, qry.stems);
      f_lmds_1500.compute(doc_entry, qry.stems);
      f_lmds_1000.compute(doc_entry, qry.stems);
      tfidf_feature.compute(doc_entry, qry.stems);
      prob_feature.compute(doc_entry, qry.stems);
      be_feature.compute(doc_entry, qry.stems);
      dph_feature.compute(doc_entry, qry.stems);
      dfr_feature.compute(doc_entry, qry.stems);
      f_stream.compute(doc_entry, qry.stems);
      features.compute(doc_entry, qry.stems);
      prox_feature.compute(doc_entry, qry);
      f_tpscore.compute(doc_entry, qry.stems);

      std::cout << label << "," << qry.id << "," << docno;
      doc_entry.present();
      std::cout << std::endl;
    }
    auto stop = clock::now();
    auto load_time =
        std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    std::cerr << "qid: " << qry.id << " in " << load_time.count() << " ms"
              << std::endl;
  }

  delete idx_cache;

  return 0;
}
