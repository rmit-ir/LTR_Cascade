#include <iostream>
#include <limits>
#include <string>

#include "CLI/CLI.hpp"
#include "cereal/archives/binary.hpp"

#include "inverted_index.hpp"
#include "forward_index.hpp"
#include "term_feature.hpp"

int main(int argc, char **argv) {
    size_t done      = 0;
    size_t freq      = 0;
    double tfidf_max = 0.0;
    double bm25_max  = 0.0;
    double pr_max    = 0.0;
    double be_max    = 0.0;
    double dfr_max   = 0.0;
    double dph_max   = 0.0;
    double lm_max    = -std::numeric_limits<double>::max();

    std::string inverted_index_file;
    std::string forward_index;
    std::string output_file;

    CLI::App app{"Unigram feature generation."};
    app.add_option("-i,--inverted-index", inverted_index_file, "Inverted index filename")->required();
    app.add_option("-f,--forward-index", forward_index, "Forward index filename")->required();
    app.add_option("-o,--out-file", output_file, "Output filename")->required();
    CLI11_PARSE(app, argc, argv);


    // load inv_idx
    std::ifstream              ifs_inv(inverted_index_file);
    cereal::BinaryInputArchive iarchive_inv(ifs_inv);
    InvertedIndex                     inv_idx;
    iarchive_inv(inv_idx);

    // load fwd_idx
    std::ifstream              ifs_fwd(forward_index);
    cereal::BinaryInputArchive iarchive_fwd(ifs_fwd);
    ForwardIndex                     fwd_idx;
    iarchive_fwd(fwd_idx);


    auto outfile = std::ofstream(output_file, std::ofstream::app);
    outfile << std::fixed << std::setprecision(6);

    size_t    clen = 0;
    size_t ndocs    = 0;
    double avg_dlen = 0.0;
    auto doclen = build_doclen(fwd_idx, clen, ndocs, avg_dlen);
    std::cout << "Avg Document Length: " << avg_dlen << std::endl;
    std::cout << "N. docs: " << ndocs << std::endl;
    std::cout << "Collection Length " << clen << std::endl;

    for(auto&& pl : inv_idx) {
        feature_t feature;
        feature.term =  pl.term;
        feature.cf = pl.totalCount;
        feature.cdf = pl.list.size();

        /* Min count is set to 4 or IQR computation goes boom. */
        if (pl.list.size() >= 4) {
            feature.geo_mean = compute_geo_mean(pl.list);
            compute_tfidf_stats(feature, doclen, pl.list, ndocs, tfidf_max);
            compute_bm25_stats(feature, doclen, pl.list, ndocs, avg_dlen, bm25_max);
            compute_lm_stats(feature, doclen, pl.list, clen, pl.totalCount, lm_max);
            compute_prob_stats(feature, doclen, pl.list, pr_max);
            compute_be_stats(feature, doclen, pl.list, ndocs, avg_dlen, pl.totalCount, be_max);
            compute_dph_stats(feature, doclen, pl.list, ndocs, avg_dlen, pl.totalCount, dph_max);
            compute_dfr_stats(feature, doclen, pl.list, ndocs, avg_dlen, pl.totalCount, dfr_max);
            outfile << feature;
            freq++;
        }
        done++;
    }
    std::cout << "Inv Lists Processed = " << done << std::endl;
    std::cout << "Inv Lists > 4 = " << freq << std::endl;
    std::cout << "TFIDF Max Score = " << tfidf_max << std::endl;
    std::cout << "BM25 Max Score = " << bm25_max << std::endl;
    std::cout << "LM Max Score = " << lm_max << std::endl;
    std::cout << "PR Max Score = " << pr_max << std::endl;
    std::cout << "BE Max Score = " << be_max << std::endl;
    std::cout << "DPH Max Score = " << dph_max << std::endl;
    std::cout << "DFR Max Score = " << dfr_max << std::endl;
    return 0;
}
