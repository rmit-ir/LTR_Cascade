#include <iostream>
#include <limits>
#include <string>

#include "CLI/CLI.hpp"
#include "cereal/archives/binary.hpp"

#include "forward_index.hpp"
#include "fgen_term.hpp"

int main(int argc, char **argv) {
    FILE *in_file = NULL;

    size_t done      = 0;
    size_t freq      = 0;
    double tfidf_max = 0.0;
    double bm25_max  = 0.0;
    double pr_max    = 0.0;
    double be_max    = 0.0;
    double dfr_max   = 0.0;
    double dph_max   = 0.0;
    double lm_max    = -std::numeric_limits<double>::max();

    uint64_t cf = 0;
    char     term[4096];
    uint64_t pos   = 0;
    uint32_t docid = 0;
    uint32_t df = 0, i = 0;

    std::string idx_file_name;
    std::string forward_index;
    std::string output_file;

    CLI::App app{"Unigram feature generation."};
    app.add_option("-i,--inv-file", idx_file_name, "Inverted file input")->required();
    app.add_option("-f,--forward-index", forward_index, "Forward index filename")->required();
    app.add_option("-o,--out-file", output_file, "Output filename")->required();
    CLI11_PARSE(app, argc, argv);

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

    if ((in_file = fopen(idx_file_name.c_str(), "rb")) == NULL) {
        std::cout << "fopen(" << idx_file_name << ")" << std::endl;
        exit(EXIT_FAILURE);
    }

    /* Walk inverted file dump from Indri. */
    while (fscanf(in_file, "%s", term) == 1) {
        feature_t feature;
        fscanf(in_file, "%" SCNu64, &cf);
        fscanf(in_file, "%" SCNu64, &pos);

        feature.cf   = cf;
        feature.cdf  = pos;
        feature.term = term;

        /* Min count is set to 4 or IQR computation goes boom. */
        if (pos >= 4) {
            std::vector<posting_t> postings;
            for (i = 0; i < pos; i++) {
                fscanf(in_file, "%" SCNu32 ":", &docid);
                fscanf(in_file, "%" SCNu32, &df);
                postings.emplace_back(docid, df);
            }
            feature.geo_mean = compute_geo_mean(postings);
            compute_tfidf_stats(feature, doclen, postings, ndocs, &tfidf_max);
            compute_bm25_stats(feature, doclen, postings, ndocs, avg_dlen, &bm25_max);
            compute_lm_stats(feature, doclen, postings, clen, cf, &lm_max);
            compute_prob_stats(feature, doclen, postings, &pr_max);
            compute_be_stats(feature, doclen, postings, ndocs, avg_dlen, cf, &be_max);
            compute_dph_stats(feature, doclen, postings, ndocs, avg_dlen, cf, &dph_max);
            compute_dfr_stats(feature, doclen, postings, ndocs, avg_dlen, cf, &dfr_max);
            outfile << feature;
            freq++;
        } else {
            for (i = 0; i < pos; i++) {
                /* Read but ignore really short lists */
                fscanf(in_file, "%" SCNu32 ":", &docid);
                fscanf(in_file, "%" SCNu32, &df);
            }
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

    fclose(in_file);
    return 0;
}
