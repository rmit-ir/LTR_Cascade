#include <iostream>
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

int main(int argc, char *argv[]) {
  int opt;
  char *qfname;            //!< query fname
  char *idx_path;          //!< indri index
  char *suffix;            //!< outfile suffix
  int w_size = -1;         //!< window size
  bool is_ordered = false; //!< ordered or not, by default u
  bool is_overlap = true;  //!< overlap or not
  //!< turn on indri-like counter,
  //!< but still in a slightly different way of counting ow.
  bool indri_like = false;
  //
  while ((opt = getopt(argc, argv, "i:q:w:uovgs:")) != -1) {
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
    case 's':
      if (!optarg) {
        std::cerr << "Missing suffix. Quit." << std::endl;
        exit(EXIT_FAILURE);
      }
      suffix = optarg;
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
  //!< prepare the output file
  std::string outfname = "";
  if (is_ordered)
    outfname += "o";
  else
    outfname += "u";
  outfname += std::to_string(w_scanner.w_size());
  outfname += suffix;
  outfname += ".txt";
  std::ofstream fout(outfname.c_str());
  //!< load query set
  std::vector<std::vector<std::string>> qry_set =
      FUtils::get_tokens(qfname, ",");

  // bigrams already done
  std::map<bigram, bool> bigram_seen;

  //!<start iterating over queries
  for (std::vector<std::vector<std::string>>::iterator qset_iter =
           qry_set.begin();
       qset_iter != qry_set.end(); qset_iter++) {
    std::string qry_str = "";
    std::vector<std::string> curr_qry = uniq_terms(*qset_iter, indri_env);
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

      std::cerr << "query: ";
      for (auto &s : curr_qry) {
        std::cerr << s << " ";
      }
      std::cerr << std::endl << "bigrams: " << qry_bigrams.size() << std::endl;

      for (auto &curr_bigram : qry_bigrams) {
        bigram term_bigram(curr_idx->term(curr_bigram.first),
            curr_idx->term(curr_bigram.second));

        std::map<bigram, bool>::iterator found;
        if (is_ordered) {
          found = bigram_seen.find(term_bigram);
        } else {
          found = std::find_if(bigram_seen.begin(), bigram_seen.end(),
              [&](const std::pair<bigram, bool>& el) {
              return (el.first.term_a == term_bigram.term_a
                && el.first.term_b == term_bigram.term_b)
              || (el.first.term_a == term_bigram.term_b &&
                el.first.term_b == term_bigram.term_a);
              });
        }

        if (found == bigram_seen.end()) {
          std::vector<indri::index::DocListIterator*> doc_iters(2);
          uint64_t min_df = tot_doc;
          int min_term = -1;
          for (int i = 0; i < 2; ++i) {
            std::string curr_str =
                (i == 0) ? curr_bigram.first : curr_bigram.second;

            qry_str += curr_str + " ";
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
            qry_str = "";
            continue;
          }

          //!< start counting and then dumping out the results
          std::vector<std::pair<lemur::api::DOCID_T, uint64_t>>
              window_postings = w_scanner.window_count(doc_iters, min_term);
          // append to output file.
          fout << qry_str << w_scanner.collection_cnt() << " "
               << window_postings.size() << " ";
          for (auto post_iter = window_postings.begin();
               post_iter != window_postings.end(); ++post_iter) {
            fout << post_iter->first << ":" << post_iter->second;
            if (std::next(post_iter) != window_postings.end()) {
              fout << " ";
            }
          }
          fout << std::endl;
          qry_str = "";

          bigram_seen.emplace(std::pair<bigram, bool>({curr_idx->term(curr_bigram.first),
                curr_idx->term(curr_bigram.second)}, true));
          delete doc_iters[0];
          delete doc_iters[1];
        } else {
          std::cerr << "already processed (" << curr_bigram.first << ", "
                    << curr_bigram.second << ")" << std::endl;
        }
      }
    }
    w_scanner.set_wsize(w_size);
  }
  fout.close();
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
