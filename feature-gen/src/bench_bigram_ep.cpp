#include <iostream>
#include <chrono>
#include <tuple>
#include "futils.hpp"
#include "w_scanner.h"
#include <unistd.h>
#include <unordered_set>
#include "indri/Repository.hpp"
#include "indri/QueryEnvironment.hpp"
#include "indri/Index.hpp"
#include "indri/DocListIterator.hpp"

using namespace indri::api;

indri::collection::Repository repository;

std::vector<std::string> uniq_terms(const std::vector<std::string> &qry,
                                    QueryEnvironment &qry_env);

struct bigram {
  int term_a;
  int term_b;

  bigram(int a, int b) : term_a(a), term_b(b) {}

  bool operator<(const bigram &rhs) const {
    return std::tie(term_a, term_b) < std::tie(rhs.term_a, rhs.term_b);
  }
};

struct bigram_posting {
  uint64_t collection_count;
  uint64_t posting_count;
  std::vector<std::pair<uint64_t, uint64_t>> postings;
};

int main(int argc, char *argv[]) {
  int opt;
  char *qfname;            //!< query fname
  char *idx_path;          //!< indri index
  int w_size = -1;         //!< window size
  bool is_ordered = false; //!< ordered or not, by default u
  bool is_overlap = true;  //!< overlap or not
  //!< turn on indri-like counter,
  //!< but still in a slightly different way of counting ow.
  bool indri_like = false;
  //
  while ((opt = getopt(argc, argv, "i:q:w:uovg")) != -1) {
    switch (opt) {
    case 'q':
      if (!optarg) {
        std::cerr << "Need query file. Quit." << std::endl;
        exit(EXIT_FAILURE);
      }
      qfname = optarg;
      break;
    case 'i':
      if (!optarg) {
        std::cerr << "Need indri index. Quit." << std::endl;
        exit(EXIT_FAILURE);
      }
      idx_path = optarg;
      break;
    case 'w':
      if (!optarg) {
        std::cerr << "Missing window size, set to default" << std::endl;
      } else {
        w_size = atoi(optarg);
      }
      break;
    case 'u':
      is_ordered = false;
      break;
    case 'o':
      is_ordered = true;
      break;
    case 'v':
      is_overlap = true;
      break;
    case 'g':
      indri_like = true;
      break;
    default:
      std::cerr << "Usage: -i <idx_path> -q <qry_file> -w <size>"
                << " -u|o [unordered|ordered] -v [overlap]"
                << " -g [use indri style]" << std::endl;
      exit(EXIT_FAILURE);
    }
  }
  if (w_size == -1) {
    std::cerr << "Window size is smaller than qlen, use 4*qlen by default."
              << std::endl;
  }
  //<! open read index
  repository.openRead(idx_path);
  indri::collection::Repository::index_state curr_state = repository.indexes();
  indri::index::Index *curr_idx = (*curr_state)[0];
  uint64_t tot_doc = curr_idx->documentCount();
  std::cerr << "Open Index, containing: " << tot_doc << " docs\n";
  //!< prepare query environment, for stemming
  QueryEnvironment indri_env;
  indri_env.addIndex(idx_path);
  //!< init scanner
  WScanner w_scanner = WScanner(w_size, indri_like, is_ordered, is_overlap);

  //!< load query set
  std::vector<std::vector<std::string>> qry_set =
      FUtils::get_tokens(qfname, ",");

  // bigrams already done
  std::map<bigram, bool> bigram_seen;

  using clock = std::chrono::high_resolution_clock;
  std::chrono::time_point<clock> start, stop;
  std::chrono::nanoseconds time;
  std::vector<uint64_t> execs;

  for (std::vector<std::vector<std::string>>::iterator qset_iter =
           qry_set.begin();
       qset_iter != qry_set.end(); qset_iter++) {
    std::string qry_str = "";
    std::vector<std::string> curr_qry = uniq_terms(*qset_iter, indri_env);
    std::vector<bigram_posting> curr_qry_postings;

    start = clock::now();

    if (curr_qry.size() < 2) {
      std::cerr << "Omit one term query." << std::endl;
    } else {
      // bigrams of query terms
      std::vector<std::pair<std::string, std::string>> qry_bigrams;
      for (size_t i = 0; i < curr_qry.size(); ++i) {
        for (size_t j = 0; j < curr_qry.size(); ++j) {
          if (i == j)
            continue;
          qry_bigrams.push_back(std::make_pair(curr_qry[i], curr_qry[j]));
        }
      }

      for (auto &curr_bigram : qry_bigrams) {
        bigram term_bigram(curr_idx->term(curr_bigram.first),
                           curr_idx->term(curr_bigram.second));
        if (bigram_seen.find(term_bigram) == bigram_seen.end()) {
          std::vector<indri::index::DocListIterator *> doc_iters(2);
          uint64_t min_df = tot_doc;
          int min_term = -1;
          for (int i = 0; i < 2; ++i) {
            std::string curr_str =
                (i == 0) ? curr_bigram.first : curr_bigram.second;

            // get inverted list iterator and start with the term has smallest
            // df
            uint64_t curr_df = curr_idx->documentCount(curr_str);
            doc_iters[i] = curr_idx->docListIterator(curr_str);
            if (!doc_iters[i]) {
              std::cerr << "no doc iterator for term " << curr_str
                        << " in bigram '" << curr_bigram.first << " "
                        << curr_bigram.second << "'" << std::endl;
              break;
            }
            doc_iters[i]->startIteration();
            if (curr_df <= min_df) {
              min_term = i;
              min_df = curr_df;
            }
          }

          if (!doc_iters[0] || !doc_iters[1]) {
            continue;
          }

          //!< start counting and then dumping out the results
          std::vector<std::pair<lemur::api::DOCID_T, uint64_t>>
              window_postings = w_scanner.window_count(doc_iters, min_term);

          struct bigram_posting ep;
          ep.collection_count = w_scanner.collection_cnt();
          ep.posting_count = window_postings.size();

          for (auto post_iter = window_postings.begin();
               post_iter != window_postings.end(); ++post_iter) {
            ep.postings.push_back(
                std::make_pair(post_iter->first, post_iter->second));
          }

          bigram_seen.emplace(
              std::pair<bigram, bool>({curr_idx->term(curr_bigram.first),
                                       curr_idx->term(curr_bigram.second)},
                                      true));

          curr_qry_postings.push_back(ep);
        } else {
          std::cerr << "already processed (" << curr_bigram.first << ", "
                    << curr_bigram.second << ")" << std::endl;
        }
      }
    }
    w_scanner.set_wsize(w_size);
    curr_qry_postings.clear();
    stop = clock::now();
    time = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);
    execs.push_back(time.count());
  }

  std::sort(execs.begin(), execs.end());
  size_t median;
  median = execs.size() / 2;
  std::cout << execs[0] << " " << execs[median] << " "
            << execs[execs.size() - 1] << std::endl;

  return 0;
}

/**
 * remove duplicate terms. based on stemming and the
 * term id. Duplicate terms are not allowed for now.
 * @param qry
 * @param qry_env. QueryEnvironment, stemmer
 * @return
 */
std::vector<std::string> uniq_terms(const std::vector<std::string> &qry,
                                    QueryEnvironment &qry_env) {
  std::vector<std::string> qry_tokens;        //!< original order must be kept
  std::unordered_set<std::string> uniq_terms; //!< deduplicate
  for (size_t i = 0; i < qry.size(); ++i) {
    std::string curr_token = qry_env.stemTerm(qry[i]);
    if (uniq_terms.empty() || uniq_terms.find(curr_token) == uniq_terms.end()) {
      //            std::cout<<curr_token<<std::endl;
      qry_tokens.push_back(curr_token);
      uniq_terms.insert(curr_token);
    }
  }
  return qry_tokens;
}
