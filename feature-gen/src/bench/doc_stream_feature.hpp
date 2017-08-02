#ifndef DOC_STREAM_FEATURE_HPP
#define DOC_STREAM_FEATURE_HPP

#include <limits>

#include "doc_feature.hpp"

namespace bench {

class doc_stream_feature : public doc_feature {

public:
  struct field_stats {
    std::string name;
    int length;
    int count;
    int tf;
    int sum_sqrs;
    int max;
    int min;
    double avg;

    field_stats()
        : name(""), length(0), count(0), tf(0), sum_sqrs(0), max(0),
          min(std::numeric_limits<int>::max()), avg(0) {}
  };

  // sum of stream len normalised by term frequency
  double _sum_stream_len;
  double _sum_stream_len_body;
  double _sum_stream_len_title;
  double _sum_stream_len_heading;
  double _sum_stream_len_inlink;
  double _sum_stream_len_a;

  // min of stream len normalised by term frequency
  double _min_stream_len;
  double _min_stream_len_body;
  double _min_stream_len_title;
  double _min_stream_len_heading;
  double _min_stream_len_inlink;
  double _min_stream_len_a;

  // max of stream len normalised by term frequency
  double _max_stream_len;
  double _max_stream_len_body;
  double _max_stream_len_title;
  double _max_stream_len_heading;
  double _max_stream_len_inlink;
  double _max_stream_len_a;

  // mean of stream len normalised by term frequency
  double _mean_stream_len;
  double _mean_stream_len_body;
  double _mean_stream_len_title;
  double _mean_stream_len_heading;
  double _mean_stream_len_inlink;
  double _mean_stream_len_a;

  // variance of stream len normalised by term frequency
  double _variance_stream_len;
  double _variance_stream_len_body;
  double _variance_stream_len_title;
  double _variance_stream_len_heading;
  double _variance_stream_len_inlink;
  double _variance_stream_len_a;

  /**
   * Constructor
   */
  doc_stream_feature(indri_index &i) : doc_feature(i) {}

  /**
   * Calculate scores
   */
  void compute(fat_cache_entry &doc, std::vector<std::string> &query_stems,
               std::string field_str = "") {
    _score_reset();

    const indri::index::TermList *term_list = doc.term_list;

    std::map<uint64_t, uint32_t> counts;
    auto fields = term_list->fields();
    field_stats curr;
    int field_id = index.field(field_str);

    curr.name = field_str;

    if (field_id < 1) {
      // field is not indexed
      return;
    }

    // reset query term counts
    for (auto &s : query_stems) {
      // a missing term has key `0`
      counts[index.term(s)] = 0;
    }

    auto const &doc_terms = term_list->terms();
    for (auto &f : fields) {
      if (f.id != static_cast<size_t>(field_id)) {
        continue;
      }

      // stream length
      curr.length += f.end - f.begin;
      // number of times field occurs
      ++curr.count;
      // max, min, sum of squares
      if (curr.min > curr.length) {
        curr.min = curr.length;
      }
      if (curr.max < curr.length) {
        curr.max = curr.length;
      }
      curr.sum_sqrs += curr.length * curr.length;

      for (size_t i = f.begin; i < f.end; ++i) {
        auto it = counts.find(doc_terms[i]);
        if (it != counts.end()) {
          ++it->second;
          ++curr.tf;
        }
      }
    }

    // average
    curr.avg = (double)curr.length / curr.count;

    // penalise docs with more than 1 title tag
    if (_field_title == curr.name && curr.count > 1) {
      curr.length = -curr.length;
    }

    _accumulate_score(field_str, curr.length);
    _set_stream_stats(field_str, curr);

    // stream length is set for the score member variables
    doc.stream_len = _score_doc;
    doc.stream_len_body = _score_body;
    doc.stream_len_title = _score_title;
    doc.stream_len_heading = _score_heading;
    doc.stream_len_inlink = _score_inlink;
    doc.stream_len_a = _score_a;

    // additional variables local to this subclass

    // sum of stream length normalised term frequency
    doc.sum_stream_len = _sum_stream_len;
    doc.sum_stream_len_body = _sum_stream_len_body;
    doc.sum_stream_len_title = _sum_stream_len_title;
    doc.sum_stream_len_heading = _sum_stream_len_heading;
    doc.sum_stream_len_inlink = _sum_stream_len_inlink;
    doc.sum_stream_len_a = _sum_stream_len_a;

    doc.min_stream_len = _min_stream_len;
    doc.min_stream_len_body = _min_stream_len_body;
    doc.min_stream_len_title = _min_stream_len_title;
    doc.min_stream_len_heading = _min_stream_len_heading;
    doc.min_stream_len_inlink = _min_stream_len_inlink;
    doc.min_stream_len_a = _min_stream_len_a;

    doc.max_stream_len = _max_stream_len;
    doc.max_stream_len_body = _max_stream_len_body;
    doc.max_stream_len_title = _max_stream_len_title;
    doc.max_stream_len_heading = _max_stream_len_heading;
    doc.max_stream_len_inlink = _max_stream_len_inlink;
    doc.max_stream_len_a = _max_stream_len_a;

    doc.mean_stream_len = _mean_stream_len;
    doc.mean_stream_len_body = _mean_stream_len_body;
    doc.mean_stream_len_title = _mean_stream_len_title;
    doc.mean_stream_len_heading = _mean_stream_len_heading;
    doc.mean_stream_len_inlink = _mean_stream_len_inlink;
    doc.mean_stream_len_a = _mean_stream_len_a;

    doc.variance_stream_len = _variance_stream_len;
    doc.variance_stream_len_body = _variance_stream_len_body;
    doc.variance_stream_len_title = _variance_stream_len_title;
    doc.variance_stream_len_heading = _variance_stream_len_heading;
    doc.variance_stream_len_inlink = _variance_stream_len_inlink;
    doc.variance_stream_len_a = _variance_stream_len_a;
  }

  /**
   * Set stats for a type of field.
   */
  void _set_stream_stats(std::string key, field_stats stats) {
    if (!stats.length || !stats.tf) {
      return;
    }

    double tmp_sum_len = stats.length;
    double tmp_min = stats.min;
    double tmp_max = stats.max;
    double tmp_mean = stats.avg;
    double tmp_variance =
        ((double)stats.sum_sqrs / stats.length) - stats.avg * stats.avg;

    tmp_sum_len /= (double)stats.tf;
    tmp_min /= (double)stats.tf;
    tmp_max /= (double)stats.tf;
    tmp_mean /= (double)stats.tf;
    tmp_variance /= (double)stats.tf;

    if (0 == key.compare(_fields[0])) {
      _sum_stream_len = tmp_sum_len;
      _min_stream_len = tmp_min;
      _max_stream_len = tmp_max;
      _mean_stream_len = tmp_mean;
      _variance_stream_len = tmp_variance;
    } else if (0 == key.compare(_fields[1])) {
      _sum_stream_len_body = tmp_sum_len;
      _min_stream_len_body = tmp_min;
      _max_stream_len_body = tmp_max;
      _mean_stream_len_body = tmp_mean;
      _variance_stream_len_body = tmp_variance;
    } else if (0 == key.compare(_fields[2])) {
      _sum_stream_len_title = tmp_sum_len;
      _min_stream_len_title = tmp_min;
      _max_stream_len_title = tmp_max;
      _mean_stream_len_title = tmp_mean;
      _variance_stream_len_title = tmp_variance;
    } else if (0 == key.compare(_fields[3])) {
      _sum_stream_len_heading = tmp_sum_len;
      _min_stream_len_heading = tmp_min;
      _max_stream_len_heading = tmp_max;
      _mean_stream_len_heading = tmp_mean;
      _variance_stream_len_heading = tmp_variance;
    } else if (0 == key.compare(_fields[4])) {
      _sum_stream_len_inlink = tmp_sum_len;
      _min_stream_len_inlink = tmp_min;
      _max_stream_len_inlink = tmp_max;
      _mean_stream_len_inlink = tmp_mean;
      _variance_stream_len_inlink = tmp_variance;
    } else if (0 == key.compare(_fields[5])) {
      _sum_stream_len_a = tmp_sum_len;
      _min_stream_len_a = tmp_min;
      _max_stream_len_a = tmp_max;
      _mean_stream_len_a = tmp_mean;
      _variance_stream_len_a = tmp_variance;
    }
  }
};

} /* bench */

#endif
