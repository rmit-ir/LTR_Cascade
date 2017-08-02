#ifndef DOCMETA_URL_FEATURE_HPP
#define DOCMETA_URL_FEATURE_HPP

#include <iostream>
#include <string>
#include <utility>

#include "fat.hpp"

class docmeta_url_feature {
  std::string url;

public:
  /**
   * Count number of slashes in url.
   */
  int url_slash_count() {
    int count = 0;
    const std::string proto = "://";
    const std::string param_delim = "?";
    size_t pos = url.find(proto);
    size_t pos_q = url.find(param_delim);

    if (pos_q < pos || std::string::npos == pos) {
      pos = 0;
    } else {
      pos += proto.size();
    }

    while (std::string::npos != (pos = url.find("/", pos + 1, 1))) {
      ++count;
    }

    return count;
  }

  void compute(fat_cache_entry &doc) {
    doc.url_slash_count = url_slash_count();
    doc.url_length = url.size();
  }

  void set_url(std::string &str) { url = str; }
};

#endif
