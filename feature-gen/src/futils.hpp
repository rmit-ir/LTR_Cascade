//
// Created by Xiaolu on 25/12/16.
//
#ifndef DUMP_NGRAM_FUTILS_HPP
#define DUMP_NGRAM_FUTILS_HPP

#include <fstream>
#include <iostream>
#include <string>
#include <cstring>
#include <vector>
#include <stdlib.h>
#include <stdio.h>

class FUtils {

public:
  FUtils() = default;
  virtual ~FUtils(){};

  /**
   * load query file
   * @param fname
   * @param delim, delimeter for tokens
   * @return vector of tokens
   */
  static std::vector<std::vector<std::string>> get_tokens(std::string fname,
                                                          std::string delim) {
    std::ifstream fin(fname.c_str(), std::ifstream::in);
    if (!fin.good()) {
      std::cerr << "From FUtils: cannot read file: " + fname << std::endl;
      exit(EXIT_FAILURE);
    }
    std::string query_str = "";
    std::vector<std::vector<std::string>> query_set;
    while (!fin.eof()) {
      std::getline(fin, query_str);
      if (query_str != "") {
        query_set.push_back(get_qres_tuple(query_str, delim));
      }
    }
    fin.close();
    return query_set;
  }

private:
  /**
   * split per query
   * @param qstr
   * @param delim
   * @return query tokens
   */
  static std::vector<std::string> get_qres_tuple(std::string qstr,
                                                 std::string delim) {
    char *cstr = new char[qstr.length() + 1];
    std::strcpy(cstr, qstr.c_str());
    // cstr now contains a c-string copy of st
    char *p = std::strtok(cstr, delim.c_str());
    std::vector<std::string> tokens;
    while (p != 0) {
      std::string curr_str(p);
      //            std::cout << curr_str << '\n';
      tokens.push_back(p);
      p = std::strtok(NULL, delim.c_str());
    }
    delete[] cstr;
    return tokens;
  }
};

#endif // DUMP_NGRAM_FUTILS_HPP
