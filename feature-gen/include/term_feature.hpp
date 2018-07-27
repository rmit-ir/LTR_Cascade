#pragma once

#include <inttypes.h>

#include <string.h>

#include <cmath>
#include <cstdio>

#include "features/bm25/bm25.hpp"
#include "features/bose_einstein/be.hpp"
#include "features/dfr/dfr.hpp"
#include "features/dph/dph.hpp"
#include "features/lmds/lm.hpp"
#include "features/probability/prob.hpp"
#include "features/tfidf/tfidf.hpp"

namespace {
constexpr double zeta = 1.960;
}

struct feature_t {
    std::string term;
    uint64_t    cf;
    uint64_t    cdf;
    double      geo_mean;
    double      tfidf_median;
    double      tfidf_first;
    double      tfidf_third;
    double      tfidf_min;
    double      tfidf_max;
    double      tfidf_avg;
    double      tfidf_variance;
    double      tfidf_std_dev;
    double      tfidf_confidence;
    double      tfidf_hmean;
    double      bm25_median;
    double      bm25_first;
    double      bm25_third;
    double      bm25_min;
    double      bm25_max;
    double      bm25_avg;
    double      bm25_variance;
    double      bm25_std_dev;
    double      bm25_confidence;
    double      bm25_hmean;
    double      lm_median;
    double      lm_first;
    double      lm_third;
    double      lm_min;
    double      lm_max;
    double      lm_avg;
    double      lm_variance;
    double      lm_std_dev;
    double      lm_confidence;
    double      lm_hmean;
    double      dfr_median;
    double      dfr_first;
    double      dfr_third;
    double      dfr_min;
    double      dfr_max;
    double      dfr_avg;
    double      dfr_variance;
    double      dfr_std_dev;
    double      dfr_confidence;
    double      dfr_hmean;
    double      dph_median;
    double      dph_first;
    double      dph_third;
    double      dph_min;
    double      dph_max;
    double      dph_avg;
    double      dph_variance;
    double      dph_std_dev;
    double      dph_confidence;
    double      dph_hmean;
    double      be_median;
    double      be_first;
    double      be_third;
    double      be_min;
    double      be_max;
    double      be_avg;
    double      be_variance;
    double      be_std_dev;
    double      be_confidence;
    double      be_hmean;
    double      pr_median;
    double      pr_first;
    double      pr_third;
    double      pr_min;
    double      pr_max;
    double      pr_avg;
    double      pr_variance;
    double      pr_std_dev;
    double      pr_confidence;
    double      pr_hmean;

    friend std::ostream &operator<<(std::ostream &os, const feature_t &f);
};

std::ostream &operator<<(std::ostream &os, const feature_t &f) {
    os << f.term << " ";
    os << f.cf << " ";
    os << f.cdf << " ";
    os << f.geo_mean << " ";
    os << f.bm25_median << " ";
    os << f.bm25_first << " ";
    os << f.bm25_third << " ";
    os << f.bm25_max << " ";
    os << f.bm25_min << " ";
    os << f.bm25_avg << " ";
    os << f.bm25_variance << " ";
    os << f.bm25_std_dev << " ";
    os << f.bm25_confidence << " ";
    os << f.bm25_hmean << " ";
    os << f.tfidf_median << " ";
    os << f.tfidf_first << " ";
    os << f.tfidf_third << " ";
    os << f.tfidf_max << " ";
    os << f.tfidf_min << " ";
    os << f.tfidf_avg << " ";
    os << f.tfidf_variance << " ";
    os << f.tfidf_std_dev << " ";
    os << f.tfidf_confidence << " ";
    os << f.tfidf_hmean << " ";
    os << f.lm_median << " ";
    os << f.lm_first << " ";
    os << f.lm_third << " ";
    os << f.lm_max << " ";
    os << f.lm_min << " ";
    os << f.lm_avg << " ";
    os << f.lm_variance << " ";
    os << f.lm_std_dev << " ";
    os << f.lm_confidence << " ";
    os << f.lm_hmean << " ";
    os << f.pr_median << " ";
    os << f.pr_first << " ";
    os << f.pr_third << " ";
    os << f.pr_max << " ";
    os << f.pr_min << " ";
    os << f.pr_avg << " ";
    os << f.pr_variance << " ";
    os << f.pr_std_dev << " ";
    os << f.pr_confidence << " ";
    os << f.pr_hmean << " ";
    os << f.be_median << " ";
    os << f.be_first << " ";
    os << f.be_third << " ";
    os << f.be_max << " ";
    os << f.be_min << " ";
    os << f.be_avg << " ";
    os << f.be_variance << " ";
    os << f.be_std_dev << " ";
    os << f.be_confidence << " ";
    os << f.be_hmean << " ";
    os << f.dph_median << " ";
    os << f.dph_first << " ";
    os << f.dph_third << " ";
    os << f.dph_max << " ";
    os << f.dph_min << " ";
    os << f.dph_avg << " ";
    os << f.dph_variance << " ";
    os << f.dph_std_dev << " ";
    os << f.dph_confidence << " ";
    os << f.dph_hmean << " ";
    os << f.dfr_median << " ";
    os << f.dfr_first << " ";
    os << f.dfr_third << " ";
    os << f.dfr_max << " ";
    os << f.dfr_min << " ";
    os << f.dfr_avg << " ";
    os << f.dfr_variance << " ";
    os << f.dfr_std_dev << " ";
    os << f.dfr_confidence << " ";
    os << f.dfr_hmean << " ";
    os << std::endl;
    return os;
}

std::vector<size_t> build_doclen(ForwardIndex &fwd, size_t &clen, size_t &ndocs, double &avg_dlen) {
    std::vector<size_t> dlen;
    for (auto &&d : fwd) {
        clen += d.length();
        dlen.push_back(d.length());
        ++ndocs;
    }
    --ndocs;
    avg_dlen = (double)clen / ndocs;
    return dlen;
}

double compute_geo_mean(const std::map<uint32_t, uint32_t> &posting) {
    double sum = 0.0;
    for (auto &&p : posting) {
        sum += p.second;
    }
    return pow(sum, (1.0 / posting.size()));
}

void compute_prob_stats(feature_t &                               f,
                        const std::vector<size_t> &               doclen,
                        const std::map<uint32_t, uint32_t> &posting,
                        double &                                  max) {
    uint32_t            size = posting.size();
    uint32_t            mid  = size / 2;
    uint32_t            lq   = size / 4;
    uint32_t            uq   = 3 * size / 4;
    double              sum = 0.0, sum_sqrs = 0.0, variance = 0.0;
    double              hmsum = 0.0;
    std::vector<double> bmtmp;

    for (const auto &pair : posting) {
        double score = calculate_prob(pair.second, doclen[pair.first]);
        bmtmp.push_back(score);
        if (score > max)
            max = score;
    }

    std::sort(bmtmp.begin(), bmtmp.end(), std::greater<double>());

    f.pr_median = size % 2 == 0 ? (bmtmp[mid] + bmtmp[mid - 1]) / 2 : bmtmp[mid];
    f.pr_first  = size % 2 == 0 ? (bmtmp[lq] + bmtmp[lq - 1]) / 2 : bmtmp[lq];
    f.pr_third  = size % 2 == 0 ? (bmtmp[uq] + bmtmp[uq - 1]) / 2 : bmtmp[uq];
    f.pr_max    = bmtmp[0];
    f.pr_min    = bmtmp[size - 1];
    for (size_t i = 0; i < size; i++) {
        sum += bmtmp[i];
        sum_sqrs += bmtmp[i] * bmtmp[i];
        hmsum += 1 / bmtmp[i];
    }

    f.pr_avg        = sum / (double)size;
    f.pr_variance   = (sum_sqrs / (double)size) - f.pr_avg * f.pr_avg;
    f.pr_std_dev    = sqrt(variance);
    f.pr_confidence = zeta * (f.pr_std_dev / (sqrt(size)));
    f.pr_hmean      = (double)size / hmsum;
}

void compute_be_stats(feature_t &                               f,
                      const std::vector<size_t> &               doclen,
                      const std::map<uint32_t, uint32_t> &posting,
                      uint64_t                                  ndocs,
                      double                                    avg_dlen,
                      uint64_t                                  c_f,
                      double &                                  max) {
    uint32_t            size = posting.size();
    uint32_t            mid  = size / 2;
    uint32_t            lq   = size / 4;
    uint32_t            uq   = 3 * size / 4;
    double              sum = 0.0, sum_sqrs = 0.0, variance = 0.0;
    double              hmsum = 0.0;
    std::vector<double> bmtmp;

    for (const auto &pair : posting) {
        double score = calculate_be(pair.second, c_f, ndocs, avg_dlen, doclen[pair.first]);
        bmtmp.push_back(score);
        if (score > max)
            max = score;
    }

    std::sort(bmtmp.begin(), bmtmp.end(), std::greater<double>());

    f.be_median = size % 2 == 0 ? (bmtmp[mid] + bmtmp[mid - 1]) / 2 : bmtmp[mid];
    f.be_first  = size % 2 == 0 ? (bmtmp[lq] + bmtmp[lq - 1]) / 2 : bmtmp[lq];
    f.be_third  = size % 2 == 0 ? (bmtmp[uq] + bmtmp[uq - 1]) / 2 : bmtmp[uq];
    f.be_max    = bmtmp[0];
    f.be_min    = bmtmp[size - 1];
    for (size_t i = 0; i < size; i++) {
        sum += bmtmp[i];
        sum_sqrs += bmtmp[i] * bmtmp[i];
        hmsum += 1 / bmtmp[i];
    }

    f.be_avg        = sum / (double)size;
    f.be_variance   = (sum_sqrs / (double)size) - f.be_avg * f.be_avg;
    f.be_std_dev    = sqrt(variance);
    f.be_confidence = zeta * (f.be_std_dev / (sqrt(size)));
    f.be_hmean      = (double)size / hmsum;
}

void compute_dph_stats(feature_t &                               f,
                       const std::vector<size_t> &               doclen,
                       const std::map<uint32_t, uint32_t> &posting,
                       uint64_t                                  ndocs,
                       double                                    avg_dlen,
                       uint64_t                                  c_f,
                       double &                                  max) {
    uint32_t            size = posting.size();
    uint32_t            mid  = size / 2;
    uint32_t            lq   = size / 4;
    uint32_t            uq   = 3 * size / 4;
    double              sum = 0.0, sum_sqrs = 0.0, variance = 0.0;
    double              hmsum = 0.0;
    std::vector<double> bmtmp;

    for (const auto &pair : posting) {
        double score = calculate_dph(pair.second, c_f, ndocs, avg_dlen, doclen[pair.first]);
        bmtmp.push_back(score);
        if (score > max)
            max = score;
    }

    std::sort(bmtmp.begin(), bmtmp.end(), std::greater<double>());

    f.dph_median = size % 2 == 0 ? (bmtmp[mid] + bmtmp[mid - 1]) / 2 : bmtmp[mid];
    f.dph_first  = size % 2 == 0 ? (bmtmp[lq] + bmtmp[lq - 1]) / 2 : bmtmp[lq];
    f.dph_third  = size % 2 == 0 ? (bmtmp[uq] + bmtmp[uq - 1]) / 2 : bmtmp[uq];
    f.dph_max    = bmtmp[0];
    f.dph_min    = bmtmp[size - 1];
    for (size_t i = 0; i < size; i++) {
        sum += bmtmp[i];
        sum_sqrs += bmtmp[i] * bmtmp[i];
        hmsum += 1 / bmtmp[i];
    }

    f.dph_avg        = sum / (double)size;
    f.dph_variance   = (sum_sqrs / (double)size) - f.dph_avg * f.dph_avg;
    f.dph_std_dev    = sqrt(variance);
    f.dph_confidence = zeta * (f.dph_std_dev / (sqrt(size)));
    f.dph_hmean      = (double)size / hmsum;
}

void compute_dfr_stats(feature_t &                               f,
                       const std::vector<size_t> &               doclen,
                       const std::map<uint32_t, uint32_t> &posting,
                       uint64_t                                  ndocs,
                       double                                    avg_dlen,
                       uint64_t                                  c_f,
                       double &                                  max) {
    uint32_t            size = posting.size();
    uint32_t            mid  = size / 2;
    uint32_t            lq   = size / 4;
    uint32_t            uq   = 3 * size / 4;
    double              sum = 0.0, sum_sqrs = 0.0, variance = 0.0;
    double              hmsum = 0.0;
    std::vector<double> bmtmp;

    for (const auto &pair : posting) {
        double score =
            calculate_dfr(pair.second, c_f, size, ndocs, avg_dlen, doclen[pair.first]);
        bmtmp.push_back(score);
        if (score > max)
            max = score;
    }

    std::sort(bmtmp.begin(), bmtmp.end(), std::greater<double>());

    f.dfr_median = size % 2 == 0 ? (bmtmp[mid] + bmtmp[mid - 1]) / 2 : bmtmp[mid];
    f.dfr_first  = size % 2 == 0 ? (bmtmp[lq] + bmtmp[lq - 1]) / 2 : bmtmp[lq];
    f.dfr_third  = size % 2 == 0 ? (bmtmp[uq] + bmtmp[uq - 1]) / 2 : bmtmp[uq];
    f.dfr_max    = bmtmp[0];
    f.dfr_min    = bmtmp[size - 1];
    for (size_t i = 0; i < size; i++) {
        sum += bmtmp[i];
        sum_sqrs += bmtmp[i] * bmtmp[i];
        hmsum += 1 / bmtmp[i];
    }

    f.dfr_avg        = sum / (double)size;
    f.dfr_variance   = (sum_sqrs / (double)size) - f.dfr_avg * f.dfr_avg;
    f.dfr_std_dev    = sqrt(variance);
    f.dfr_confidence = zeta * (f.dfr_std_dev / (sqrt(size)));
    f.dfr_hmean      = (double)size / hmsum;
}

void compute_tfidf_stats(feature_t &                               f,
                         const std::vector<size_t> &               doclen,
                         const std::map<uint32_t, uint32_t> &posting,
                         uint64_t                                  ndocs,
                         double &                                  max) {
    size_t              size = posting.size();
    uint32_t            mid  = size / 2;
    uint32_t            lq   = size / 4;
    uint32_t            uq   = 3 * size / 4;
    double              sum = 0.0, sum_sqrs = 0.0, variance = 0.0;
    double              hmsum = 0.0;
    std::vector<double> bmtmp;

    for (const auto &pair : posting) {
        double score = calculate_tfidf(pair.second, size, doclen[pair.first], ndocs);
        bmtmp.push_back(score);
        if (score > max)
            max = score;
    }

    std::sort(bmtmp.begin(), bmtmp.end(), std::greater<double>());

    f.tfidf_median = size % 2 == 0 ? (bmtmp[mid] + bmtmp[mid - 1]) / 2 : bmtmp[mid];
    f.tfidf_first  = size % 2 == 0 ? (bmtmp[lq] + bmtmp[lq - 1]) / 2 : bmtmp[lq];
    f.tfidf_third  = size % 2 == 0 ? (bmtmp[uq] + bmtmp[uq - 1]) / 2 : bmtmp[uq];
    f.tfidf_max    = bmtmp[0];
    f.tfidf_min    = bmtmp[size - 1];
    for (size_t i = 0; i < size; i++) {
        sum += bmtmp[i];
        sum_sqrs += bmtmp[i] * bmtmp[i];
        hmsum += 1 / bmtmp[i];
    }

    f.tfidf_avg        = sum / (double)size;
    f.tfidf_variance   = (sum_sqrs / (double)size) - f.tfidf_avg * f.tfidf_avg;
    f.tfidf_std_dev    = sqrt(variance);
    f.tfidf_confidence = zeta * (f.tfidf_std_dev / (sqrt(size)));
    f.tfidf_hmean      = (double)size / hmsum;
}

void compute_bm25_stats(feature_t &                               f,
                        const std::vector<size_t> &               doclen,
                        const std::map<uint32_t, uint32_t> &posting,
                        uint64_t                                  ndocs,
                        double                                    avg_dlen,
                        double &                                  max) {
    uint32_t            size = posting.size();
    uint32_t            mid  = size / 2;
    uint32_t            lq   = size / 4;
    uint32_t            uq   = 3 * size / 4;
    double              sum = 0.0, sum_sqrs = 0.0, variance = 0.0;
    double              hmsum = 0.0;
    std::vector<double> bmtmp;

    rank_bm25 ranker;
    ranker.set_k1(90);
    ranker.set_b(40);
    ranker.num_docs    = ndocs;
    ranker.avg_doc_len = avg_dlen;

    for (const auto &pair : posting) {
        double score = ranker.calculate_docscore(1, pair.second, size, doclen[pair.first]);
        bmtmp.push_back(score);
        if (score > max)
            max = score;
    }

    std::sort(bmtmp.begin(), bmtmp.end(), std::greater<double>());

    f.bm25_median = size % 2 == 0 ? (bmtmp[mid] + bmtmp[mid - 1]) / 2 : bmtmp[mid];
    f.bm25_first  = size % 2 == 0 ? (bmtmp[lq] + bmtmp[lq - 1]) / 2 : bmtmp[lq];
    f.bm25_third  = size % 2 == 0 ? (bmtmp[uq] + bmtmp[uq - 1]) / 2 : bmtmp[uq];
    f.bm25_max    = bmtmp[0];
    f.bm25_min    = bmtmp[size - 1];
    for (size_t i = 0; i < size; i++) {
        sum += bmtmp[i];
        sum_sqrs += bmtmp[i] * bmtmp[i];
        hmsum += 1 / bmtmp[i];
    }

    f.bm25_avg        = sum / (double)size;
    f.bm25_variance   = (sum_sqrs / (double)size) - f.bm25_avg * f.bm25_avg;
    f.bm25_std_dev    = sqrt(variance);
    f.bm25_confidence = zeta * (f.bm25_std_dev / (sqrt(size)));
    f.bm25_hmean      = (double)size / hmsum;
}

void compute_lm_stats(feature_t &                               f,
                      const std::vector<size_t> &               doclen,
                      const std::map<uint32_t, uint32_t> &posting,
                      uint64_t                                  clen,
                      uint64_t                                  cf,
                      double &                                  max) {
    uint32_t            size = posting.size();
    uint32_t            mid  = size / 2;
    uint32_t            lq   = size / 4;
    uint32_t            uq   = 3 * size / 4;
    double              sum = 0.0, sum_sqrs = 0.0, variance = 0.0;
    double              hmsum = 0.0;
    double              mu    = 2500.00;
    std::vector<double> bmtmp;

    for (const auto &pair : posting) {
        double score = calculate_lm(pair.second, cf, doclen[pair.first], clen, mu);
        bmtmp.push_back(score);
        if (score > max)
            max = score;
    }

    std::sort(bmtmp.begin(), bmtmp.end(), std::greater<double>());

    f.lm_median = size % 2 == 0 ? (bmtmp[mid] + bmtmp[mid - 1]) / 2 : bmtmp[mid];
    f.lm_first  = size % 2 == 0 ? (bmtmp[lq] + bmtmp[lq - 1]) / 2 : bmtmp[lq];
    f.lm_third  = size % 2 == 0 ? (bmtmp[uq] + bmtmp[uq - 1]) / 2 : bmtmp[uq];
    f.lm_max    = bmtmp[0];
    f.lm_min    = bmtmp[size - 1];
    for (size_t i = 0; i < size; i++) {
        sum += bmtmp[i];
        sum_sqrs += bmtmp[i] * bmtmp[i];
        hmsum += 1 / bmtmp[i];
    }

    f.lm_avg        = sum / (double)size;
    f.lm_variance   = (sum_sqrs / (double)size) - f.lm_avg * f.lm_avg;
    f.lm_std_dev    = sqrt(variance);
    f.lm_confidence = zeta * (f.lm_std_dev / (sqrt(size)));
    f.lm_hmean      = (double)size / hmsum;
}
