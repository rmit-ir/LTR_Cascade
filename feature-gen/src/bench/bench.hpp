#ifndef BENCH_HPP
#define BENCH_HPP

#include <cstdint>
#include <vector>

struct bench_entry {
  int qid;
  size_t qlen;

  std::vector<uint64_t> execs_be_doc;
  std::vector<uint64_t> execs_be_body;
  std::vector<uint64_t> execs_be_title;
  std::vector<uint64_t> execs_be_heading;
  std::vector<uint64_t> execs_be_inlink;
  std::vector<uint64_t> execs_be_a;
  std::vector<uint64_t> execs_bm25_doc;
  std::vector<uint64_t> execs_bm25_body;
  std::vector<uint64_t> execs_bm25_title;
  std::vector<uint64_t> execs_bm25_heading;
  std::vector<uint64_t> execs_bm25_inlink;
  std::vector<uint64_t> execs_bm25_a;
  std::vector<uint64_t> execs_tpscore_doc;
  std::vector<uint64_t> execs_tp_interval_w100_doc;
  std::vector<uint64_t> execs_cdf_bigram_u8_doc;
  std::vector<uint64_t> execs_dfr_doc;
  std::vector<uint64_t> execs_dfr_body;
  std::vector<uint64_t> execs_dfr_title;
  std::vector<uint64_t> execs_dfr_heading;
  std::vector<uint64_t> execs_dfr_inlink;
  std::vector<uint64_t> execs_dfr_a;
  std::vector<uint64_t> execs_dph_doc;
  std::vector<uint64_t> execs_dph_body;
  std::vector<uint64_t> execs_dph_title;
  std::vector<uint64_t> execs_dph_heading;
  std::vector<uint64_t> execs_dph_inlink;
  std::vector<uint64_t> execs_dph_a;
  std::vector<uint64_t> execs_lmds_doc;
  std::vector<uint64_t> execs_lmds_body;
  std::vector<uint64_t> execs_lmds_title;
  std::vector<uint64_t> execs_lmds_heading;
  std::vector<uint64_t> execs_lmds_inlink;
  std::vector<uint64_t> execs_lmds_a;
  std::vector<uint64_t> execs_prob_doc;
  std::vector<uint64_t> execs_prob_body;
  std::vector<uint64_t> execs_prob_title;
  std::vector<uint64_t> execs_prob_heading;
  std::vector<uint64_t> execs_prob_inlink;
  std::vector<uint64_t> execs_prob_a;
  std::vector<uint64_t> execs_stream_doc;
  std::vector<uint64_t> execs_stream_body;
  std::vector<uint64_t> execs_stream_title;
  std::vector<uint64_t> execs_stream_heading;
  std::vector<uint64_t> execs_stream_inlink;
  std::vector<uint64_t> execs_stream_a;
  std::vector<uint64_t> execs_tfidf_doc;
  std::vector<uint64_t> execs_tfidf_body;
  std::vector<uint64_t> execs_tfidf_title;
  std::vector<uint64_t> execs_tfidf_heading;
  std::vector<uint64_t> execs_tfidf_inlink;
  std::vector<uint64_t> execs_tfidf_a;
  std::vector<uint64_t> execs_qry_tag_cnt_doc;
  std::vector<uint64_t> execs_qry_tag_cnt_title;
  std::vector<uint64_t> execs_qry_tag_cnt_heading;
  std::vector<uint64_t> execs_qry_tag_cnt_inlink;

  bench_entry() {}
};

#endif
