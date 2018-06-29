//
// Created by Xiaolu on 25/12/16.
//

#ifndef DUMP_NGRAM_WINDOW_SCANNER_H
#define DUMP_NGRAM_WINDOW_SCANNER_H

#include <utility>
#include <vector>
#include "lemur/IndexTypes.hpp"
#include "indri/DocListIterator.hpp"

/**
 * for combine cdf use
 */
struct TermPos {
  TermPos() {
    t_pos = -1;
    t_idx = -1;
  }
  TermPos(int idx, int pos) : t_idx(idx), t_pos(pos){};
  TermPos(const TermPos &another) {
    t_idx = another.t_idx;
    t_pos = another.t_pos;
  }
  void update(const TermPos &another) {
    t_idx = another.t_idx;
    t_pos = another.t_pos;
  }

  int t_idx;
  int t_pos;

  bool operator<(const TermPos &another) const { return t_pos < another.t_pos; }
  bool operator>(const TermPos &another) const { return t_pos > another.t_pos; }
};

class WScanner {

public:
  //!< ctrs
  WScanner();
  WScanner(int w_size, bool indri_style = false, bool is_ordered = false,
           bool is_overlap = true);
  WScanner(const WScanner &another);
  virtual ~WScanner(){};

  //!< scanning func
  /**
   * count the unordered window
   * @param doc_iters
   * @return inv file
   */
  std::vector<std::pair<lemur::api::DOCID_T, uint64_t>>
  window_count(std::vector<indri::index::DocListIterator *> &doc_iters,
               size_t min_term);

  //!< for check
  int w_size() const;
  bool is_ordered() const;
  bool is_overlap() const;
  uint64_t collection_cnt() const;
  bool indri_like() const;

  //
  /**
   * need to update window size, otherwise
   * things will go *wrong*
   * @param size, window size
   */
  void set_wsize(int size);

protected:
  /**
   * scan over cdf to count
   * @param cdf. *vector* of position list.
   * @return
   */
  uint64_t _get_uwindows(std::vector<TermPos> &cdf, size_t qlen);
  /**
   * get ordered window, same process, but need to additionally
   * get the order same as in the original query
   * @param cdf
   * @param qlen
   * @return
   */
  uint64_t _get_owindows(std::vector<TermPos> &cdf, size_t qlen);

private:
  int _w_size;
  bool _is_ordered;
  bool _is_overlap;
  bool _indri_like;
  uint64_t _collection_cnt;
};

#endif // DUMP_NGRAM_WINDOW_SCANNER_H
