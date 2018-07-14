#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <ratio>
#include <string>

#include "indri_bridge.hpp"
#include "fat.hpp"
#include "query_train_file.hpp"
#include "trec_run_file.hpp"

#include "document_features.hpp"
#include "doc_bm25_atire_feature.hpp"
#include "doc_lm_dir_feature.hpp"
#include "doc_tfidf_feature.hpp"
#include "doc_prob_feature.hpp"
#include "doc_be_feature.hpp"
#include "doc_dph_feature.hpp"
#include "doc_dfr_feature.hpp"
#include "doc_stream_feature.hpp"
#include "doc_tpscore_feature.hpp"
#include "doc_proximity_feature.hpp"
#include "bench.hpp"

static void display_usage(char *name) {
  std::cerr << "usage: " << name
            << " <query file> <trec file> <indri repository>" << std::endl;
  exit(EXIT_FAILURE);
}

static uint64_t bench_run(bench::doc_feature &f, fat_cache_entry &doc,
                          query_train &q) {

  using clock = std::chrono::high_resolution_clock;

  std::chrono::time_point<clock> start, stop;
  std::chrono::nanoseconds time;

  start = clock::now();
  f.compute(doc, q.stems);
  stop = clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);

  return time.count();
}

static uint64_t bench_run_field(bench::doc_feature &f, fat_cache_entry &doc,
                                query_train &q, std::string const &field) {

  using clock = std::chrono::high_resolution_clock;

  std::chrono::time_point<clock> start, stop;
  std::chrono::nanoseconds time;

  start = clock::now();
  f.compute(doc, q.stems, field);
  stop = clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);

  return time.count();
}

int main(int argc, char **argv) {
  using clock = std::chrono::high_resolution_clock;

  if (argc < 4) {
    display_usage(argv[0]);
  }

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

  bench::doc_be_feature f_be(*index);
  bench::doc_bm25_atire_feature f_bm25_atire(*index);
  bench::doc_dfr_feature f_dfr(*index);
  bench::doc_dph_feature f_dph(*index);
  bench::doc_lm_dir_feature f_lmds_2500(*index);
  bench::doc_prob_feature f_prob(*index);
  bench::doc_proximity_feature f_cdf_bigram(*index);
  bench::doc_proximity_feature f_tp_interval(*index);
  bench::doc_stream_feature f_stream(*index);
  bench::doc_tfidf_feature f_tfidf(*index);
  bench::doc_tpscore_feature f_tpscore(*index);
  bench::document_features f_qry_tag_cnt(*index);

  std::vector<bench_entry> dat;
  bench_entry entry;

  for (auto &qry : qtfile.get_queries()) {
    std::vector<std::string> docnos = trec_run.get_result(qry.id);
    std::vector<docid_t> docids =
        qry_env.document_ids_from_metadata("docno", docnos);

    std::cerr << docids.size() << std::endl;
    if (!docids.size()) {
      std::cerr << "no docs for qid " << qry.id << std::endl;
      continue;
    }

    idx_cache->add_docs(docids);

    entry.qid = qry.id;
    entry.qlen = qry.stems.size();

    for (size_t i = 0; i < docids.size(); ++i) {
      auto const docid = docids[i];

      std::chrono::time_point<clock> start, stop;
      std::chrono::nanoseconds time;

      start = clock::now();
      auto it = idx_cache->get_entry(docid);
      auto &doc_entry = it->second;
      stop = clock::now();
      time = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);

      uint64_t const lt = time.count();

      entry.execs_be_doc.push_back(bench_run(f_be, doc_entry, qry) + lt);
      entry.execs_be_body.push_back(
          bench_run_field(f_be, doc_entry, qry, "body") + lt);
      entry.execs_be_title.push_back(
          bench_run_field(f_be, doc_entry, qry, "title") + lt);
      entry.execs_be_heading.push_back(
          bench_run_field(f_be, doc_entry, qry, "heading") + lt);
      entry.execs_be_inlink.push_back(
          bench_run_field(f_be, doc_entry, qry, "inlink") + lt);
      entry.execs_be_a.push_back(bench_run_field(f_be, doc_entry, qry, "a") +
                                 lt);

      entry.execs_bm25_doc.push_back(bench_run(f_bm25_atire, doc_entry, qry) +
                                     lt);
      entry.execs_bm25_body.push_back(
          bench_run_field(f_bm25_atire, doc_entry, qry, "body") + lt);
      entry.execs_bm25_title.push_back(
          bench_run_field(f_bm25_atire, doc_entry, qry, "title") + lt);
      entry.execs_bm25_heading.push_back(
          bench_run_field(f_bm25_atire, doc_entry, qry, "heading") + lt);
      entry.execs_bm25_inlink.push_back(
          bench_run_field(f_bm25_atire, doc_entry, qry, "inlink") + lt);
      entry.execs_bm25_a.push_back(
          bench_run_field(f_bm25_atire, doc_entry, qry, "a") + lt);

      entry.execs_dfr_doc.push_back(bench_run(f_dfr, doc_entry, qry) + lt);
      entry.execs_dfr_body.push_back(
          bench_run_field(f_dfr, doc_entry, qry, "body") + lt);
      entry.execs_dfr_title.push_back(
          bench_run_field(f_dfr, doc_entry, qry, "title") + lt);
      entry.execs_dfr_heading.push_back(
          bench_run_field(f_dfr, doc_entry, qry, "heading") + lt);
      entry.execs_dfr_inlink.push_back(
          bench_run_field(f_dfr, doc_entry, qry, "inlink") + lt);
      entry.execs_dfr_a.push_back(bench_run_field(f_dfr, doc_entry, qry, "a") +
                                  lt);

      entry.execs_dph_doc.push_back(bench_run(f_dph, doc_entry, qry) + lt);
      entry.execs_dph_body.push_back(
          bench_run_field(f_dph, doc_entry, qry, "body") + lt);
      entry.execs_dph_title.push_back(
          bench_run_field(f_dph, doc_entry, qry, "title") + lt);
      entry.execs_dph_heading.push_back(
          bench_run_field(f_dph, doc_entry, qry, "heading") + lt);
      entry.execs_dph_inlink.push_back(
          bench_run_field(f_dph, doc_entry, qry, "inlink") + lt);
      entry.execs_dph_a.push_back(bench_run_field(f_dph, doc_entry, qry, "a") +
                                  lt);

      entry.execs_lmds_doc.push_back(bench_run(f_lmds_2500, doc_entry, qry) +
                                     lt);
      entry.execs_lmds_body.push_back(
          bench_run_field(f_lmds_2500, doc_entry, qry, "body") + lt);
      entry.execs_lmds_title.push_back(
          bench_run_field(f_lmds_2500, doc_entry, qry, "title") + lt);
      entry.execs_lmds_heading.push_back(
          bench_run_field(f_lmds_2500, doc_entry, qry, "heading") + lt);
      entry.execs_lmds_inlink.push_back(
          bench_run_field(f_lmds_2500, doc_entry, qry, "inlink") + lt);
      entry.execs_lmds_a.push_back(
          bench_run_field(f_lmds_2500, doc_entry, qry, "a") + lt);

      entry.execs_prob_doc.push_back(bench_run(f_prob, doc_entry, qry) + lt);
      entry.execs_prob_body.push_back(
          bench_run_field(f_prob, doc_entry, qry, "body") + lt);
      entry.execs_prob_title.push_back(
          bench_run_field(f_prob, doc_entry, qry, "title") + lt);
      entry.execs_prob_heading.push_back(
          bench_run_field(f_prob, doc_entry, qry, "heading") + lt);
      entry.execs_prob_inlink.push_back(
          bench_run_field(f_prob, doc_entry, qry, "inlink") + lt);
      entry.execs_prob_a.push_back(
          bench_run_field(f_prob, doc_entry, qry, "a") + lt);

      entry.execs_stream_doc.push_back(bench_run(f_stream, doc_entry, qry) +
                                       lt);
      entry.execs_stream_body.push_back(
          bench_run_field(f_stream, doc_entry, qry, "body") + lt);
      entry.execs_stream_title.push_back(
          bench_run_field(f_stream, doc_entry, qry, "title") + lt);
      entry.execs_stream_heading.push_back(
          bench_run_field(f_stream, doc_entry, qry, "heading") + lt);
      entry.execs_stream_inlink.push_back(
          bench_run_field(f_stream, doc_entry, qry, "inlink") + lt);
      entry.execs_stream_a.push_back(
          bench_run_field(f_stream, doc_entry, qry, "a") + lt);

      entry.execs_tfidf_doc.push_back(bench_run(f_tfidf, doc_entry, qry) + lt);
      entry.execs_tfidf_body.push_back(
          bench_run_field(f_tfidf, doc_entry, qry, "body") + lt);
      entry.execs_tfidf_title.push_back(
          bench_run_field(f_tfidf, doc_entry, qry, "title") + lt);
      entry.execs_tfidf_heading.push_back(
          bench_run_field(f_tfidf, doc_entry, qry, "heading") + lt);
      entry.execs_tfidf_inlink.push_back(
          bench_run_field(f_tfidf, doc_entry, qry, "inlink") + lt);
      entry.execs_tfidf_a.push_back(
          bench_run_field(f_tfidf, doc_entry, qry, "a") + lt);

      entry.execs_tpscore_doc.push_back(bench_run(f_tpscore, doc_entry, qry) +
                                        lt);

      // proximity are a different class
      start = clock::now();
      f_tp_interval.compute_tp_interval(doc_entry, qry);
      stop = clock::now();
      time = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);
      entry.execs_tp_interval_w100_doc.push_back(time.count() + lt);

      start = clock::now();
      f_cdf_bigram.compute_cdf_bigram(doc_entry, qry);
      stop = clock::now();
      time = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);
      entry.execs_cdf_bigram_u8_doc.push_back(time.count() + lt);

      // document features is a different class
      start = clock::now();
      f_qry_tag_cnt.compute(doc_entry, qry.stems);
      stop = clock::now();
      time = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);
      entry.execs_qry_tag_cnt_doc.push_back(time.count() + lt);

      start = clock::now();
      f_qry_tag_cnt.compute(doc_entry, qry.stems, "title");
      stop = clock::now();
      time = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);
      entry.execs_qry_tag_cnt_title.push_back(time.count() + lt);

      start = clock::now();
      f_qry_tag_cnt.compute(doc_entry, qry.stems, "heading");
      stop = clock::now();
      time = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);
      entry.execs_qry_tag_cnt_heading.push_back(time.count() + lt);

      start = clock::now();
      f_qry_tag_cnt.compute(doc_entry, qry.stems, "inlink");
      stop = clock::now();
      time = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);
      entry.execs_qry_tag_cnt_inlink.push_back(time.count() + lt);
    }
  }

  std::sort(entry.execs_be_doc.begin(), entry.execs_be_doc.end());
  std::sort(entry.execs_be_body.begin(), entry.execs_be_body.end());
  std::sort(entry.execs_be_title.begin(), entry.execs_be_title.end());
  std::sort(entry.execs_be_heading.begin(), entry.execs_be_heading.end());
  std::sort(entry.execs_be_inlink.begin(), entry.execs_be_inlink.end());
  std::sort(entry.execs_be_a.begin(), entry.execs_be_a.end());
  std::sort(entry.execs_bm25_doc.begin(), entry.execs_bm25_doc.end());
  std::sort(entry.execs_bm25_body.begin(), entry.execs_bm25_body.end());
  std::sort(entry.execs_bm25_title.begin(), entry.execs_bm25_title.end());
  std::sort(entry.execs_bm25_heading.begin(), entry.execs_bm25_heading.end());
  std::sort(entry.execs_bm25_inlink.begin(), entry.execs_bm25_inlink.end());
  std::sort(entry.execs_bm25_a.begin(), entry.execs_bm25_a.end());
  std::sort(entry.execs_tpscore_doc.begin(), entry.execs_tpscore_doc.end());
  std::sort(entry.execs_tp_interval_w100_doc.begin(),
            entry.execs_tp_interval_w100_doc.end());
  std::sort(entry.execs_cdf_bigram_u8_doc.begin(),
            entry.execs_cdf_bigram_u8_doc.end());
  std::sort(entry.execs_dfr_doc.begin(), entry.execs_dfr_doc.end());
  std::sort(entry.execs_dfr_body.begin(), entry.execs_dfr_body.end());
  std::sort(entry.execs_dfr_title.begin(), entry.execs_dfr_title.end());
  std::sort(entry.execs_dfr_heading.begin(), entry.execs_dfr_heading.end());
  std::sort(entry.execs_dfr_inlink.begin(), entry.execs_dfr_inlink.end());
  std::sort(entry.execs_dfr_a.begin(), entry.execs_dfr_a.end());
  std::sort(entry.execs_dph_doc.begin(), entry.execs_dph_doc.end());
  std::sort(entry.execs_dph_body.begin(), entry.execs_dph_body.end());
  std::sort(entry.execs_dph_title.begin(), entry.execs_dph_title.end());
  std::sort(entry.execs_dph_heading.begin(), entry.execs_dph_heading.end());
  std::sort(entry.execs_dph_inlink.begin(), entry.execs_dph_inlink.end());
  std::sort(entry.execs_dph_a.begin(), entry.execs_dph_a.end());
  std::sort(entry.execs_lmds_doc.begin(), entry.execs_lmds_doc.end());
  std::sort(entry.execs_lmds_body.begin(), entry.execs_lmds_body.end());
  std::sort(entry.execs_lmds_title.begin(), entry.execs_lmds_title.end());
  std::sort(entry.execs_lmds_heading.begin(), entry.execs_lmds_heading.end());
  std::sort(entry.execs_lmds_inlink.begin(), entry.execs_lmds_inlink.end());
  std::sort(entry.execs_lmds_a.begin(), entry.execs_lmds_a.end());
  std::sort(entry.execs_prob_doc.begin(), entry.execs_prob_doc.end());
  std::sort(entry.execs_prob_body.begin(), entry.execs_prob_body.end());
  std::sort(entry.execs_prob_title.begin(), entry.execs_prob_title.end());
  std::sort(entry.execs_prob_heading.begin(), entry.execs_prob_heading.end());
  std::sort(entry.execs_prob_inlink.begin(), entry.execs_prob_inlink.end());
  std::sort(entry.execs_prob_a.begin(), entry.execs_prob_a.end());
  std::sort(entry.execs_stream_doc.begin(), entry.execs_stream_doc.end());
  std::sort(entry.execs_stream_body.begin(), entry.execs_stream_body.end());
  std::sort(entry.execs_stream_title.begin(), entry.execs_stream_title.end());
  std::sort(entry.execs_stream_heading.begin(),
            entry.execs_stream_heading.end());
  std::sort(entry.execs_stream_inlink.begin(), entry.execs_stream_inlink.end());
  std::sort(entry.execs_stream_a.begin(), entry.execs_stream_a.end());
  std::sort(entry.execs_tfidf_doc.begin(), entry.execs_tfidf_doc.end());
  std::sort(entry.execs_tfidf_body.begin(), entry.execs_tfidf_body.end());
  std::sort(entry.execs_tfidf_title.begin(), entry.execs_tfidf_title.end());
  std::sort(entry.execs_tfidf_heading.begin(), entry.execs_tfidf_heading.end());
  std::sort(entry.execs_tfidf_inlink.begin(), entry.execs_tfidf_inlink.end());
  std::sort(entry.execs_tfidf_a.begin(), entry.execs_tfidf_a.end());
  std::sort(entry.execs_qry_tag_cnt_doc.begin(),
            entry.execs_qry_tag_cnt_doc.end());
  std::sort(entry.execs_qry_tag_cnt_title.begin(),
            entry.execs_qry_tag_cnt_title.end());
  std::sort(entry.execs_qry_tag_cnt_heading.begin(),
            entry.execs_qry_tag_cnt_heading.end());
  std::sort(entry.execs_qry_tag_cnt_inlink.begin(),
            entry.execs_qry_tag_cnt_inlink.end());

  size_t median;

  median = entry.execs_be_doc.size() / 2;
  std::cout << entry.execs_be_doc[0] << " " << entry.execs_be_doc[median] << " "
            << entry.execs_be_doc[entry.execs_be_doc.size() - 1] << std::endl;
  median = entry.execs_be_body.size() / 2;
  std::cout << entry.execs_be_body[0] << " " << entry.execs_be_body[median]
            << " " << entry.execs_be_body[entry.execs_be_body.size() - 1]
            << std::endl;
  median = entry.execs_be_title.size() / 2;
  std::cout << entry.execs_be_title[0] << " " << entry.execs_be_title[median]
            << " " << entry.execs_be_title[entry.execs_be_title.size() - 1]
            << std::endl;
  median = entry.execs_be_heading.size() / 2;
  std::cout << entry.execs_be_heading[0] << " "
            << entry.execs_be_heading[median] << " "
            << entry.execs_be_heading[entry.execs_be_heading.size() - 1]
            << std::endl;
  median = entry.execs_be_inlink.size() / 2;
  std::cout << entry.execs_be_inlink[0] << " " << entry.execs_be_inlink[median]
            << " " << entry.execs_be_inlink[entry.execs_be_inlink.size() - 1]
            << std::endl;
  median = entry.execs_be_a.size() / 2;
  std::cout << entry.execs_be_a[0] << " " << entry.execs_be_a[median] << " "
            << entry.execs_be_a[entry.execs_be_a.size() - 1] << std::endl;
  median = entry.execs_bm25_doc.size() / 2;
  std::cout << entry.execs_bm25_doc[0] << " " << entry.execs_bm25_doc[median]
            << " " << entry.execs_bm25_doc[entry.execs_bm25_doc.size() - 1]
            << std::endl;
  median = entry.execs_bm25_body.size() / 2;
  std::cout << entry.execs_bm25_body[0] << " " << entry.execs_bm25_body[median]
            << " " << entry.execs_bm25_body[entry.execs_bm25_body.size() - 1]
            << std::endl;
  median = entry.execs_bm25_title.size() / 2;
  std::cout << entry.execs_bm25_title[0] << " "
            << entry.execs_bm25_title[median] << " "
            << entry.execs_bm25_title[entry.execs_bm25_title.size() - 1]
            << std::endl;
  median = entry.execs_bm25_heading.size() / 2;
  std::cout << entry.execs_bm25_heading[0] << " "
            << entry.execs_bm25_heading[median] << " "
            << entry.execs_bm25_heading[entry.execs_bm25_heading.size() - 1]
            << std::endl;
  median = entry.execs_bm25_inlink.size() / 2;
  std::cout << entry.execs_bm25_inlink[0] << " "
            << entry.execs_bm25_inlink[median] << " "
            << entry.execs_bm25_inlink[entry.execs_bm25_inlink.size() - 1]
            << std::endl;
  median = entry.execs_bm25_a.size() / 2;
  std::cout << entry.execs_bm25_a[0] << " " << entry.execs_bm25_a[median] << " "
            << entry.execs_bm25_a[entry.execs_bm25_a.size() - 1] << std::endl;
  median = entry.execs_tpscore_doc.size() / 2;
  std::cout << entry.execs_tpscore_doc[0] << " "
            << entry.execs_tpscore_doc[median] << " "
            << entry.execs_tpscore_doc[entry.execs_tpscore_doc.size() - 1]
            << std::endl;
  median = entry.execs_tp_interval_w100_doc.size() / 2;
  std::cout << entry.execs_tp_interval_w100_doc[0] << " "
            << entry.execs_tp_interval_w100_doc[median] << " "
            << entry.execs_tp_interval_w100_doc
                   [entry.execs_tp_interval_w100_doc.size() - 1]
            << std::endl;
  median = entry.execs_cdf_bigram_u8_doc.size() / 2;
  std::cout
      << entry.execs_cdf_bigram_u8_doc[0] << " "
      << entry.execs_cdf_bigram_u8_doc[median] << " "
      << entry.execs_cdf_bigram_u8_doc[entry.execs_cdf_bigram_u8_doc.size() - 1]
      << std::endl;
  median = entry.execs_dfr_doc.size() / 2;
  std::cout << entry.execs_dfr_doc[0] << " " << entry.execs_dfr_doc[median]
            << " " << entry.execs_dfr_doc[entry.execs_dfr_doc.size() - 1]
            << std::endl;
  median = entry.execs_dfr_body.size() / 2;
  std::cout << entry.execs_dfr_body[0] << " " << entry.execs_dfr_body[median]
            << " " << entry.execs_dfr_body[entry.execs_dfr_body.size() - 1]
            << std::endl;
  median = entry.execs_dfr_title.size() / 2;
  std::cout << entry.execs_dfr_title[0] << " " << entry.execs_dfr_title[median]
            << " " << entry.execs_dfr_title[entry.execs_dfr_title.size() - 1]
            << std::endl;
  median = entry.execs_dfr_heading.size() / 2;
  std::cout << entry.execs_dfr_heading[0] << " "
            << entry.execs_dfr_heading[median] << " "
            << entry.execs_dfr_heading[entry.execs_dfr_heading.size() - 1]
            << std::endl;
  median = entry.execs_dfr_inlink.size() / 2;
  std::cout << entry.execs_dfr_inlink[0] << " "
            << entry.execs_dfr_inlink[median] << " "
            << entry.execs_dfr_inlink[entry.execs_dfr_inlink.size() - 1]
            << std::endl;
  median = entry.execs_dfr_a.size() / 2;
  std::cout << entry.execs_dfr_a[0] << " " << entry.execs_dfr_a[median] << " "
            << entry.execs_dfr_a[entry.execs_dfr_a.size() - 1] << std::endl;
  median = entry.execs_dph_doc.size() / 2;
  std::cout << entry.execs_dph_doc[0] << " " << entry.execs_dph_doc[median]
            << " " << entry.execs_dph_doc[entry.execs_dph_doc.size() - 1]
            << std::endl;
  median = entry.execs_dph_body.size() / 2;
  std::cout << entry.execs_dph_body[0] << " " << entry.execs_dph_body[median]
            << " " << entry.execs_dph_body[entry.execs_dph_body.size() - 1]
            << std::endl;
  median = entry.execs_dph_title.size() / 2;
  std::cout << entry.execs_dph_title[0] << " " << entry.execs_dph_title[median]
            << " " << entry.execs_dph_title[entry.execs_dph_title.size() - 1]
            << std::endl;
  median = entry.execs_dph_heading.size() / 2;
  std::cout << entry.execs_dph_heading[0] << " "
            << entry.execs_dph_heading[median] << " "
            << entry.execs_dph_heading[entry.execs_dph_heading.size() - 1]
            << std::endl;
  median = entry.execs_dph_inlink.size() / 2;
  std::cout << entry.execs_dph_inlink[0] << " "
            << entry.execs_dph_inlink[median] << " "
            << entry.execs_dph_inlink[entry.execs_dph_inlink.size() - 1]
            << std::endl;
  median = entry.execs_dph_a.size() / 2;
  std::cout << entry.execs_dph_a[0] << " " << entry.execs_dph_a[median] << " "
            << entry.execs_dph_a[entry.execs_dph_a.size() - 1] << std::endl;
  median = entry.execs_lmds_doc.size() / 2;
  std::cout << entry.execs_lmds_doc[0] << " " << entry.execs_lmds_doc[median]
            << " " << entry.execs_lmds_doc[entry.execs_lmds_doc.size() - 1]
            << std::endl;
  median = entry.execs_lmds_body.size() / 2;
  std::cout << entry.execs_lmds_body[0] << " " << entry.execs_lmds_body[median]
            << " " << entry.execs_lmds_body[entry.execs_lmds_body.size() - 1]
            << std::endl;
  median = entry.execs_lmds_title.size() / 2;
  std::cout << entry.execs_lmds_title[0] << " "
            << entry.execs_lmds_title[median] << " "
            << entry.execs_lmds_title[entry.execs_lmds_title.size() - 1]
            << std::endl;
  median = entry.execs_lmds_heading.size() / 2;
  std::cout << entry.execs_lmds_heading[0] << " "
            << entry.execs_lmds_heading[median] << " "
            << entry.execs_lmds_heading[entry.execs_lmds_heading.size() - 1]
            << std::endl;
  median = entry.execs_lmds_inlink.size() / 2;
  std::cout << entry.execs_lmds_inlink[0] << " "
            << entry.execs_lmds_inlink[median] << " "
            << entry.execs_lmds_inlink[entry.execs_lmds_inlink.size() - 1]
            << std::endl;
  median = entry.execs_lmds_a.size() / 2;
  std::cout << entry.execs_lmds_a[0] << " " << entry.execs_lmds_a[median] << " "
            << entry.execs_lmds_a[entry.execs_lmds_a.size() - 1] << std::endl;
  median = entry.execs_prob_doc.size() / 2;
  std::cout << entry.execs_prob_doc[0] << " " << entry.execs_prob_doc[median]
            << " " << entry.execs_prob_doc[entry.execs_prob_doc.size() - 1]
            << std::endl;
  median = entry.execs_prob_body.size() / 2;
  std::cout << entry.execs_prob_body[0] << " " << entry.execs_prob_body[median]
            << " " << entry.execs_prob_body[entry.execs_prob_body.size() - 1]
            << std::endl;
  median = entry.execs_prob_title.size() / 2;
  std::cout << entry.execs_prob_title[0] << " "
            << entry.execs_prob_title[median] << " "
            << entry.execs_prob_title[entry.execs_prob_title.size() - 1]
            << std::endl;
  median = entry.execs_prob_heading.size() / 2;
  std::cout << entry.execs_prob_heading[0] << " "
            << entry.execs_prob_heading[median] << " "
            << entry.execs_prob_heading[entry.execs_prob_heading.size() - 1]
            << std::endl;
  median = entry.execs_prob_inlink.size() / 2;
  std::cout << entry.execs_prob_inlink[0] << " "
            << entry.execs_prob_inlink[median] << " "
            << entry.execs_prob_inlink[entry.execs_prob_inlink.size() - 1]
            << std::endl;
  median = entry.execs_prob_a.size() / 2;
  std::cout << entry.execs_prob_a[0] << " " << entry.execs_prob_a[median] << " "
            << entry.execs_prob_a[entry.execs_prob_a.size() - 1] << std::endl;
  median = entry.execs_stream_doc.size() / 2;
  std::cout << entry.execs_stream_doc[0] << " "
            << entry.execs_stream_doc[median] << " "
            << entry.execs_stream_doc[entry.execs_stream_doc.size() - 1]
            << std::endl;
  median = entry.execs_stream_body.size() / 2;
  std::cout << entry.execs_stream_body[0] << " "
            << entry.execs_stream_body[median] << " "
            << entry.execs_stream_body[entry.execs_stream_body.size() - 1]
            << std::endl;
  median = entry.execs_stream_title.size() / 2;
  std::cout << entry.execs_stream_title[0] << " "
            << entry.execs_stream_title[median] << " "
            << entry.execs_stream_title[entry.execs_stream_title.size() - 1]
            << std::endl;
  median = entry.execs_stream_heading.size() / 2;
  std::cout << entry.execs_stream_heading[0] << " "
            << entry.execs_stream_heading[median] << " "
            << entry.execs_stream_heading[entry.execs_stream_heading.size() - 1]
            << std::endl;
  median = entry.execs_stream_inlink.size() / 2;
  std::cout << entry.execs_stream_inlink[0] << " "
            << entry.execs_stream_inlink[median] << " "
            << entry.execs_stream_inlink[entry.execs_stream_inlink.size() - 1]
            << std::endl;
  median = entry.execs_stream_a.size() / 2;
  std::cout << entry.execs_stream_a[0] << " " << entry.execs_stream_a[median]
            << " " << entry.execs_stream_a[entry.execs_stream_a.size() - 1]
            << std::endl;
  median = entry.execs_tfidf_doc.size() / 2;
  std::cout << entry.execs_tfidf_doc[0] << " " << entry.execs_tfidf_doc[median]
            << " " << entry.execs_tfidf_doc[entry.execs_tfidf_doc.size() - 1]
            << std::endl;
  median = entry.execs_tfidf_body.size() / 2;
  std::cout << entry.execs_tfidf_body[0] << " "
            << entry.execs_tfidf_body[median] << " "
            << entry.execs_tfidf_body[entry.execs_tfidf_body.size() - 1]
            << std::endl;
  median = entry.execs_tfidf_title.size() / 2;
  std::cout << entry.execs_tfidf_title[0] << " "
            << entry.execs_tfidf_title[median] << " "
            << entry.execs_tfidf_title[entry.execs_tfidf_title.size() - 1]
            << std::endl;
  median = entry.execs_tfidf_heading.size() / 2;
  std::cout << entry.execs_tfidf_heading[0] << " "
            << entry.execs_tfidf_heading[median] << " "
            << entry.execs_tfidf_heading[entry.execs_tfidf_heading.size() - 1]
            << std::endl;
  median = entry.execs_tfidf_inlink.size() / 2;
  std::cout << entry.execs_tfidf_inlink[0] << " "
            << entry.execs_tfidf_inlink[median] << " "
            << entry.execs_tfidf_inlink[entry.execs_tfidf_inlink.size() - 1]
            << std::endl;
  median = entry.execs_tfidf_a.size() / 2;
  std::cout << entry.execs_tfidf_a[0] << " " << entry.execs_tfidf_a[median]
            << " " << entry.execs_tfidf_a[entry.execs_tfidf_a.size() - 1]
            << std::endl;
  median = entry.execs_qry_tag_cnt_doc.size() / 2;
  std::cout
      << entry.execs_qry_tag_cnt_doc[0] << " "
      << entry.execs_qry_tag_cnt_doc[median] << " "
      << entry.execs_qry_tag_cnt_doc[entry.execs_qry_tag_cnt_doc.size() - 1]
      << std::endl;
  median = entry.execs_qry_tag_cnt_title.size() / 2;
  std::cout
      << entry.execs_qry_tag_cnt_title[0] << " "
      << entry.execs_qry_tag_cnt_title[median] << " "
      << entry.execs_qry_tag_cnt_title[entry.execs_qry_tag_cnt_title.size() - 1]
      << std::endl;
  median = entry.execs_qry_tag_cnt_heading.size() / 2;
  std::cout << entry.execs_qry_tag_cnt_heading[0] << " "
            << entry.execs_qry_tag_cnt_heading[median] << " "
            << entry.execs_qry_tag_cnt_heading
                   [entry.execs_qry_tag_cnt_heading.size() - 1]
            << std::endl;
  median = entry.execs_qry_tag_cnt_inlink.size() / 2;
  std::cout
      << entry.execs_qry_tag_cnt_inlink[0] << " "
      << entry.execs_qry_tag_cnt_inlink[median] << " "
      << entry.execs_qry_tag_cnt_inlink[entry.execs_qry_tag_cnt_inlink.size() -
                                        1]
      << std::endl;

  delete idx_cache;

  return 0;
}
