#ifndef FAT_HPP
#define FAT_HPP

#include <algorithm>
#include <chrono>
#include <iostream>
#include <iterator>
#include <map>
#include <unordered_map>
#include <vector>
#include <utility>

#include "indri/Repository.hpp"
#include "indri/Index.hpp"
#include "indri/TermList.hpp"
#include "indri/ScopedLock.hpp"
#include "indri/DocumentVector.hpp"
#include "indri/PriorListIterator.hpp"

struct prior_entry {
  int id = 0;
  double val = 0;

  operator double() const { return val; }
};

struct fat_cache_entry {
  enum entry_state {
    STATE_NEW = 0,
    STATE_INERT,
  };

  // doc id for convenience
  int id = 0;
  // entry state becomes inert once features are calculated as a cache hint
  size_t state = STATE_NEW;
  int length = 0;

  double pagerank = 0;
  // Score from training trec run file
  double stage0_score = 0;
  // BM25 Atire
  double bm25_atire = 0;
  double bm25_atire_body = 0;
  double bm25_atire_title = 0;
  double bm25_atire_heading = 0;
  double bm25_atire_inlink = 0;
  double bm25_atire_a = 0;
  // BM25 TREC3 k = 1.2, b = 0.75
  double bm25_trec3 = 0;
  double bm25_trec3_body = 0;
  double bm25_trec3_title = 0;
  double bm25_trec3_heading = 0;
  double bm25_trec3_inlink = 0;
  double bm25_trec3_a = 0;
  // BM25 TREC3 k = 2.0, b = 0.75
  double bm25_trec3_kmax = 0;
  double bm25_trec3_kmax_body = 0;
  double bm25_trec3_kmax_title = 0;
  double bm25_trec3_kmax_heading = 0;
  double bm25_trec3_kmax_inlink = 0;
  double bm25_trec3_kmax_a = 0;
  // QL mu = 2500
  double lm_dir_2500 = 0;
  double lm_dir_2500_body = 0;
  double lm_dir_2500_title = 0;
  double lm_dir_2500_heading = 0;
  double lm_dir_2500_inlink = 0;
  double lm_dir_2500_a = 0;
  // QL mu = 1500
  double lm_dir_1500 = 0;
  double lm_dir_1500_body = 0;
  double lm_dir_1500_title = 0;
  double lm_dir_1500_heading = 0;
  double lm_dir_1500_inlink = 0;
  double lm_dir_1500_a = 0;
  // QL mu = 1000
  double lm_dir_1000 = 0;
  double lm_dir_1000_body = 0;
  double lm_dir_1000_title = 0;
  double lm_dir_1000_heading = 0;
  double lm_dir_1000_inlink = 0;
  double lm_dir_1000_a = 0;
  // tfidf
  double tfidf = 0;
  double tfidf_body = 0;
  double tfidf_title = 0;
  double tfidf_heading = 0;
  double tfidf_inlink = 0;
  double tfidf_a = 0;
  // Probability
  double prob = 0;
  double prob_body = 0;
  double prob_title = 0;
  double prob_heading = 0;
  double prob_inlink = 0;
  double prob_a = 0;
  // DFR: Bose-Einstien
  double be = 0;
  double be_body = 0;
  double be_title = 0;
  double be_heading = 0;
  double be_inlink = 0;
  double be_a = 0;
  // DFR: DPH
  double dph = 0;
  double dph_body = 0;
  double dph_title = 0;
  double dph_heading = 0;
  double dph_inlink = 0;
  double dph_a = 0;
  // DFR: BB2
  double dfr = 0;
  double dfr_body = 0;
  double dfr_title = 0;
  double dfr_heading = 0;
  double dfr_inlink = 0;
  double dfr_a = 0;
  // Stream
  double stream_len = 0;
  double stream_len_body = 0;
  double stream_len_title = 0;
  double stream_len_heading = 0;
  double stream_len_inlink = 0;
  double stream_len_a = 0;
  // Stream sum normalised by tf
  double sum_stream_len = 0;
  double sum_stream_len_body = 0;
  double sum_stream_len_title = 0;
  double sum_stream_len_heading = 0;
  double sum_stream_len_inlink = 0;
  double sum_stream_len_a = 0;
  // Stream min normalised by tf
  double min_stream_len = 0;
  double min_stream_len_body = 0;
  double min_stream_len_title = 0;
  double min_stream_len_heading = 0;
  double min_stream_len_inlink = 0;
  double min_stream_len_a = 0;
  // Stream max normalised by tf
  double max_stream_len = 0;
  double max_stream_len_body = 0;
  double max_stream_len_title = 0;
  double max_stream_len_heading = 0;
  double max_stream_len_inlink = 0;
  double max_stream_len_a = 0;
  // Stream mean normalised by tf
  double mean_stream_len = 0;
  double mean_stream_len_body = 0;
  double mean_stream_len_title = 0;
  double mean_stream_len_heading = 0;
  double mean_stream_len_inlink = 0;
  double mean_stream_len_a = 0;
  // Stream variance normalised by tf
  double variance_stream_len = 0;
  double variance_stream_len_body = 0;
  double variance_stream_len_title = 0;
  double variance_stream_len_heading = 0;
  double variance_stream_len_inlink = 0;
  double variance_stream_len_a = 0;

  // TP-Score
  double tpscore = 0;

  // BM25 bigram unordered window score (sum of unigram scores)
  double bm25_bigram_u8 = 0;
  // BM25 score of bigram intervals in window (Lu, et al.)
  double bm25_tp_dist_w100 = 0;

  // The frequency of query terms within the <title> tag
  size_t tag_title_qry_count = 0;
  // The frequency of query terms within the <heading> tag
  size_t tag_heading_qry_count = 0;
  // The frequency of query terms within the <mainbody> tag
  size_t tag_mainbody_qry_count = 0;
  // The frequency of query terms within the inlinks
  size_t tag_inlink_qry_count = 0;

  // The number of times the <title> tag appears in the document
  int tag_title_count = 0;
  // The number of times the <heading> tag appears in the document
  int tag_heading_count = 0; // Indri heading field includes tags h1-h4
  // The number of inlinks in the document
  int tag_inlink_count = 0;
  // The number of times the <applet> tag appears in the document
  int tag_applet_count = 0;
  // The number of times the <object> tag appears in the document
  int tag_object_count = 0;
  // The number of times the <embed> tag appears in the document
  int tag_embed_count = 0;

  // Number of slashes in URL
  int url_slash_count = 0;
  // URL length
  size_t url_length = 0;

  indri::index::TermList *term_list = nullptr;
  indri::api::DocumentVector *doc_vec = nullptr;

  fat_cache_entry() {}
  fat_cache_entry(int i, double pr, indri::index::TermList *tl,
                  indri::api::DocumentVector *dv)
      : id(i), pagerank(pr), term_list(tl), doc_vec(dv) {}

  void present() {
    std::cout << "," << pagerank;

    std::cout << "," << stage0_score;

    std::cout << "," << bm25_atire;
    std::cout << "," << bm25_atire_body;
    std::cout << "," << bm25_atire_title;
    std::cout << "," << bm25_atire_heading;
    std::cout << "," << bm25_atire_inlink;
    std::cout << "," << bm25_atire_a;

    std::cout << "," << bm25_trec3;
    std::cout << "," << bm25_trec3_body;
    std::cout << "," << bm25_trec3_title;
    std::cout << "," << bm25_trec3_heading;
    std::cout << "," << bm25_trec3_inlink;
    std::cout << "," << bm25_trec3_a;

    std::cout << "," << bm25_trec3_kmax;
    std::cout << "," << bm25_trec3_kmax_body;
    std::cout << "," << bm25_trec3_kmax_title;
    std::cout << "," << bm25_trec3_kmax_heading;
    std::cout << "," << bm25_trec3_kmax_inlink;
    std::cout << "," << bm25_trec3_kmax_a;

    std::cout << "," << bm25_bigram_u8;

    std::cout << "," << bm25_tp_dist_w100;

    std::cout << "," << tpscore;

    std::cout << "," << lm_dir_2500;
    std::cout << "," << lm_dir_2500_body;
    std::cout << "," << lm_dir_2500_title;
    std::cout << "," << lm_dir_2500_heading;
    std::cout << "," << lm_dir_2500_inlink;
    std::cout << "," << lm_dir_2500_a;

    std::cout << "," << lm_dir_1500;
    std::cout << "," << lm_dir_1500_body;
    std::cout << "," << lm_dir_1500_title;
    std::cout << "," << lm_dir_1500_heading;
    std::cout << "," << lm_dir_1500_inlink;
    std::cout << "," << lm_dir_1500_a;

    std::cout << "," << lm_dir_1000;
    std::cout << "," << lm_dir_1000_body;
    std::cout << "," << lm_dir_1000_title;
    std::cout << "," << lm_dir_1000_heading;
    std::cout << "," << lm_dir_1000_inlink;
    std::cout << "," << lm_dir_1000_a;

    std::cout << "," << tfidf;
    std::cout << "," << tfidf_body;
    std::cout << "," << tfidf_title;
    std::cout << "," << tfidf_heading;
    std::cout << "," << tfidf_inlink;
    std::cout << "," << tfidf_a;

    std::cout << "," << prob;
    std::cout << "," << prob_body;
    std::cout << "," << prob_title;
    std::cout << "," << prob_heading;
    std::cout << "," << prob_inlink;
    std::cout << "," << prob_a;

    std::cout << "," << be;
    std::cout << "," << be_body;
    std::cout << "," << be_title;
    std::cout << "," << be_heading;
    std::cout << "," << be_inlink;
    std::cout << "," << be_a;

    std::cout << "," << dph;
    std::cout << "," << dph_body;
    std::cout << "," << dph_title;
    std::cout << "," << dph_heading;
    std::cout << "," << dph_inlink;
    std::cout << "," << dph_a;

    std::cout << "," << dfr;
    std::cout << "," << dfr_body;
    std::cout << "," << dfr_title;
    std::cout << "," << dfr_heading;
    std::cout << "," << dfr_inlink;
    std::cout << "," << dfr_a;

    std::cout << "," << stream_len;
    std::cout << "," << stream_len_body;
    std::cout << "," << stream_len_title;
    std::cout << "," << stream_len_heading;
    std::cout << "," << stream_len_inlink;
    std::cout << "," << stream_len_a;

    std::cout << "," << sum_stream_len;
    std::cout << "," << sum_stream_len_body;
    std::cout << "," << sum_stream_len_title;
    std::cout << "," << sum_stream_len_heading;
    std::cout << "," << sum_stream_len_inlink;
    std::cout << "," << sum_stream_len_a;

    std::cout << "," << min_stream_len;
    std::cout << "," << min_stream_len_body;
    std::cout << "," << min_stream_len_title;
    std::cout << "," << min_stream_len_heading;
    std::cout << "," << min_stream_len_inlink;
    std::cout << "," << min_stream_len_a;

    std::cout << "," << max_stream_len;
    std::cout << "," << max_stream_len_body;
    std::cout << "," << max_stream_len_title;
    std::cout << "," << max_stream_len_heading;
    std::cout << "," << max_stream_len_inlink;
    std::cout << "," << max_stream_len_a;

    std::cout << "," << mean_stream_len;
    std::cout << "," << mean_stream_len_body;
    std::cout << "," << mean_stream_len_title;
    std::cout << "," << mean_stream_len_heading;
    std::cout << "," << mean_stream_len_inlink;
    std::cout << "," << mean_stream_len_a;

    std::cout << "," << variance_stream_len;
    std::cout << "," << variance_stream_len_body;
    std::cout << "," << variance_stream_len_title;
    std::cout << "," << variance_stream_len_heading;
    std::cout << "," << variance_stream_len_inlink;
    std::cout << "," << variance_stream_len_a;

    std::cout << "," << static_cast<double>(tag_title_qry_count);
    std::cout << "," << static_cast<double>(tag_heading_qry_count);
    std::cout << "," << static_cast<double>(tag_mainbody_qry_count);
    std::cout << "," << static_cast<double>(tag_inlink_qry_count);

    std::cout << "," << static_cast<double>(tag_title_count);
    std::cout << "," << static_cast<double>(tag_heading_count);
    std::cout << "," << static_cast<double>(tag_inlink_count);
    std::cout << "," << static_cast<double>(tag_applet_count);
    std::cout << "," << static_cast<double>(tag_object_count);
    std::cout << "," << static_cast<double>(tag_embed_count);

    std::cout << "," << static_cast<double>(url_slash_count);
    std::cout << "," << static_cast<double>(url_length);
  }
};

struct fat_cache {
  static const size_t k_init_size = 16384;
  bool _init = false;
  indri::collection::Repository *repo = nullptr;
  indri::index::Index *index = nullptr;

  // PageRank for all docs in collection
  std::vector<prior_entry> prior_pagerank;
  // Term cache used by `DocumentVector` internally
  std::map<int, std::string> docvec_term_cache;

  // Cache of document data
  std::unordered_map<int, fat_cache_entry> entries;

  fat_cache(indri::collection::Repository *r, indri::index::Index *idx,
            size_t num_docs)
      : repo(r), index(idx) {
    // document id's start at 1
    prior_pagerank.resize(num_docs + 1);
  }

  ~fat_cache() {
    for (auto el : entries) {
      delete el.second.term_list;
      delete el.second.doc_vec;
    }
  }

  void init() {
    if (_init) {
      return;
    }

    _init = true;
    entries.reserve(k_init_size);

    std::cerr << "fat_cache::init...";
    std::vector<std::string> prior_names = {"pagerank"};
    std::vector<std::vector<prior_entry>> priors = {prior_pagerank};
    for (size_t i = 0; i < priors.size(); i++) {
      auto &curr_name = prior_names[i];
      std::vector<prior_entry> &curr_prior = priors[i];

      auto *it = repo->priorListIterator(curr_name);
      if (!it) {
        std::cerr << "prior '" << curr_name << "' does not exist" << std::endl;
        continue;
      }

      it->startIteration();
      while (!it->finished()) {
        auto *entry = it->currentEntry();
        if (static_cast<size_t>(entry->document) >= curr_prior.size()) {
          throw std::out_of_range("prior out of range");
        }
        curr_prior[entry->document].id = entry->document;
        curr_prior[entry->document].val = entry->score;
        it->nextEntry();
      }
      delete it;
    }
    std::cerr << " done." << std::endl;

    prior_pagerank = priors[0];
  }

  void add_docs(std::vector<int> const &vec) {
    using clock = std::chrono::high_resolution_clock;
    int docs_cached = 0;

    auto start = clock::now();

    if (entries.size() > k_init_size * 2) {
      std::cerr << "clearing cache of " << entries.size() << std::endl;
      entries.clear();
      entries.reserve(k_init_size);
    }

    // sort incoming docids
    std::vector<int> tmp(vec);
    std::sort(tmp.begin(), tmp.end());

    // cache document data
    indri::thread::ScopedLock lock(index->statisticsLock());
    for (size_t i = 0; i < tmp.size(); ++i) {
      const indri::index::TermList *term_list = index->termList(tmp[i]);
      indri::api::DocumentVector *dv =
          new indri::api::DocumentVector(index, term_list, docvec_term_cache);

      if (entries.find(tmp[i]) == entries.end()) {
        fat_cache_entry entry(tmp[i], prior_pagerank[i],
                              const_cast<indri::index::TermList *>(term_list),
                              dv);
        entry.length = index->documentLength(tmp[i]);
        entries.insert(std::make_pair(tmp[i], entry));
        ++docs_cached;
      }
    }

    auto stop = clock::now();
    auto load_time =
        std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    std::cerr << "fat_cache::add_docs cached " << docs_cached << " in "
              << load_time.count() << " ms" << std::endl;
  }

  std::unordered_map<int, fat_cache_entry>::iterator const
  get_entry(int const id) {
    // assume the entry exists
    return entries.find(id);
  }
};

#endif
