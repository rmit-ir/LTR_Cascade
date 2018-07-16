//
// Created by Xiaolu on 25/12/16.
//

#include "w_scanner.h"
#include "indri/greedy_vector"
#include <algorithm>
#include <bitset>

WScanner::WScanner() {}

WScanner::WScanner(int w_size, bool indri_like, bool is_ordered, bool is_overlap) {
    _w_size         = w_size;
    _is_ordered     = is_ordered;
    _is_overlap     = is_overlap;
    _collection_cnt = 0;
    _indri_like     = indri_like;
}

WScanner::WScanner(const WScanner &another) {
    _w_size         = another.w_size();
    _is_overlap     = another.is_overlap();
    _collection_cnt = another.collection_cnt();
    _indri_like     = another._indri_like;
}

std::vector<std::pair<lemur::api::DOCID_T, uint64_t>> WScanner::window_count(
    std::vector<indri::index::DocListIterator *> &doc_iters, size_t min_term) {
    _collection_cnt = 0;
    std::vector<std::pair<lemur::api::DOCID_T, uint64_t>> window_postings;
    if (_w_size == -1) {
        _w_size = doc_iters.size() * 4 + 1;
    }
    //!< current entry of the smallest df term
    lemur::api::DOCID_T                curr_doc = doc_iters[min_term]->currentEntry()->document;
    lemur::api::DOCID_T                max_doc  = curr_doc; //!< always keep current largest doc
    std::vector<TermPos>               position_list; //!< CDF container
    bool                               is_end   = false;
    indri::utility::greedy_vector<int> pos_list = doc_iters[min_term]->currentEntry()->positions;
    for (size_t j = 0; j < pos_list.size(); ++j) { //!< init
        position_list.push_back(TermPos(min_term, pos_list[j]));
    }
    std::make_heap(position_list.begin(), position_list.end());
    while (doc_iters[min_term]->nextEntry()) { //!< use shortest to get doc id is enough
        for (size_t i = 0; i < doc_iters.size(); ++i) {
            if (i != min_term) {
                lemur::api::DOCID_T tmp_doc = doc_iters[i]->currentEntry()->document;
                if (tmp_doc < max_doc) {
                    if (doc_iters[i]->nextEntry(max_doc)) {
                        //!< we only get geq entries
                        tmp_doc = doc_iters[i]->currentEntry()->document;
                        if (tmp_doc != curr_doc) {
                            position_list.clear();
                            max_doc = tmp_doc > max_doc ? tmp_doc : max_doc;
                            break;
                        }
                    } else {
                        position_list.clear();
                        is_end = true;
                        break;
                    }
                }
                if (tmp_doc == curr_doc) {
                    pos_list = doc_iters[i]->currentEntry()->positions;
                    for (size_t j = 0; j < pos_list.size(); ++j) {
                        position_list.push_back(TermPos(i, pos_list[j]));
                        std::push_heap(position_list.begin(), position_list.end());
                    }
                }
            }
        }
        if (is_end)
            break;
        if (!position_list.empty()) {
            //!< create CDF, by min heap
            uint64_t w_cnt = 0;
            if (!_is_ordered) {
                w_cnt = _get_uwindows(position_list, doc_iters.size());
            } else if (_is_ordered) {
                w_cnt = _get_owindows(position_list, doc_iters.size());
            }
            if (w_cnt > 0) {
                window_postings.push_back(std::make_pair(max_doc, w_cnt));
                _collection_cnt += w_cnt;
            }
        }
        position_list.clear();
        curr_doc = doc_iters[min_term]->currentEntry()->document;
        //!< move to next doc geq the curret maximum doc, since we are boolean
        if (curr_doc < max_doc) {
            if (!doc_iters[min_term]->nextEntry(max_doc)) {
                break;
            }
        }
        curr_doc = doc_iters[min_term]->currentEntry()->document;
        max_doc  = curr_doc;
        pos_list = doc_iters[min_term]->currentEntry()->positions;
        for (size_t j = 0; j < pos_list.size(); ++j) {
            position_list.push_back(TermPos(min_term, pos_list[j]));
            std::push_heap(position_list.begin(), position_list.end());
        }
    }
    return window_postings;
}

uint64_t WScanner::_get_uwindows(std::vector<TermPos> &cdf, size_t qlen) {
    uint64_t cnt = 0;
    std::sort_heap(cdf.begin(), cdf.end());
    TermPos         lhs, rhs;
    size_t          l = 0;
    size_t          r = l;
    std::bitset<32> seen; //!< track seen terms
    int             last_pos = 0;
    while (l < cdf.size()) {
        lhs = cdf[l];
        seen.set(lhs.t_idx);
        r   = l + 1;
        rhs = cdf[r];
        while (r < cdf.size() && rhs.t_pos - lhs.t_pos + 1 <= w_size()) {
            if (lhs.t_idx == rhs.t_idx && !_indri_like) //!< this is for optimal intvl
                break;
            if (!seen[rhs.t_idx]) {
                seen.set(rhs.t_idx);
            }
            if (seen.count() == qlen && (rhs.t_pos - lhs.t_pos + 1) <= w_size()) {
                cnt++;
                //                std::cout<<"(" <<lhs.t_idx<<", "<<rhs.t_idx<<")->("
                //                         <<lhs.t_pos<<", "<<rhs.t_pos<<")"<<" ";
                last_pos = r;
                break;
            }
            r++;
            rhs = cdf[r];
        }
        if (is_overlap()) {
            l++;
        } else {
            l = last_pos + 1;
            if (l >= cdf.size()) {
                break;
            }
        }
        seen.reset();
    }
    return cnt;
}

uint64_t WScanner::_get_owindows(std::vector<TermPos> &cdf, size_t qlen) {
    uint64_t cnt = 0;
    std::sort_heap(cdf.begin(), cdf.end());
    TermPos         lhs, rhs;
    size_t          l = 0;
    size_t          r = l;
    std::bitset<32> seen; //!< track seen terms
    int             last_pos = 0;
    int             last_term;
    while (l < cdf.size()) {
        lhs       = cdf[l];
        last_term = lhs.t_idx;
        seen.set(lhs.t_idx);
        r   = l + 1;
        rhs = cdf[r];
        while (r < cdf.size() && rhs.t_pos - lhs.t_pos + 1 <= w_size()) {
            if (lhs.t_idx == rhs.t_idx && !_indri_like) //!< this is for optimal intvl
                break;
            if (!seen[rhs.t_idx]) {
                if (rhs.t_idx - last_term != 1) {
                    //!< no need to continue if the terms are unordered.
                    break;
                }
                //!< only update last term when a new term comes
                last_term = rhs.t_idx;
                seen.set(rhs.t_idx);
            }
            if (seen.count() == qlen && (rhs.t_pos - lhs.t_pos + 1) <= w_size()) {
                cnt++;
                //                std::cout<<"(" <<lhs.t_idx<<", "<<rhs.t_idx<<")->("
                //                         <<lhs.t_pos<<", "<<rhs.t_pos<<")"<<std::endl;
                last_pos = r;
                break;
            }
            r++;
            rhs = cdf[r];
        }
        if (is_overlap()) {
            l++;
        } else {
            l = last_pos + 1;
            if (l >= cdf.size()) {
                break;
            }
        }
        seen.reset();
    }
    return cnt;
}

void WScanner::set_wsize(int size) { _w_size = size; }

int WScanner::w_size() const { return _w_size; }

bool WScanner::is_ordered() const { return _is_ordered; }

bool WScanner::is_overlap() const { return _is_overlap; }

uint64_t WScanner::collection_cnt() const { return _collection_cnt; }

bool WScanner::indri_like() const { return _indri_like; }
