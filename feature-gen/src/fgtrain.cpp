#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

#include "cereal/archives/binary.hpp"
#include "doc_entry.hpp"
#include "features/features.hpp"
#include "freqs_entry.hpp"

#include "query_train_file.hpp"
#include "trec_run_file.hpp"

static void display_usage(char *name) {
    std::cerr << "usage: " << name
              << " <query file> <trec file> <indri repository> <forward index> "
                 "<output file>"
              << std::endl;
    exit(EXIT_FAILURE);
}

/**
 * Generate document features and output to CSV format.
 */
int main(int argc, char **argv) {
    if (argc < 6) {
        display_usage(argv[0]);
    }

    std::string query_file         = argv[1];
    std::string trec_file          = argv[2];
    std::string repo_path          = argv[3];
    std::string forward_index_file = argv[4];

    std::string output_file = argv[5];
    auto        outfile     = std::ofstream(output_file, std::ofstream::app);
    outfile << std::fixed << std::setprecision(5);

    query_environment         indri_env;
    query_environment_adapter qry_env(&indri_env);
    qry_env.add_index(repo_path);

    indri::collection::Repository repo;
    repo.openRead(repo_path);
    auto index = (*repo.indexes())[0];

    // load query file
    std::ifstream ifs(query_file);
    if (!ifs.is_open()) {
        std::cerr << "Could not open file: " << query_file << std::endl;
        exit(EXIT_FAILURE);
    }
    query_train_file qtfile(ifs, qry_env, index);
    qtfile.parse();
    ifs.close();
    ifs.clear();

    using clock = std::chrono::high_resolution_clock;
    auto start  = clock::now();

    // load trec run file
    ifs.open(trec_file);
    trec_run_file trec_run(ifs);
    trec_run.parse();
    ifs.close();
    ifs.clear();

    // load fwd_idx
    std::ifstream              ifs_fwd(forward_index_file);
    cereal::BinaryInputArchive iarchive(ifs_fwd);
    FwdIdx                     fwd_idx;
    iarchive(fwd_idx);

    auto stop      = clock::now();
    auto load_time = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    std::cerr << "Loaded " << forward_index_file << " in " << load_time.count() << " ms"
              << std::endl;

    document_features           features(*index);
    docmeta_url_feature         meta_feature;
    doc_bm25_atire_feature      f_bm25_atire(*index);
    doc_bm25_trec3_feature      f_bm25_trec3(*index);
    doc_bm25_trec3_kmax_feature f_bm25_trec3_kmax(*index);
    doc_proximity_feature       prox_feature(*index);
    doc_lm_dir_feature          f_lmds_2500(*index);
    doc_lm_dir_1500_feature     f_lmds_1500(*index);
    doc_lm_dir_1000_feature     f_lmds_1000(*index);
    doc_tfidf_feature           tfidf_feature(*index);
    doc_prob_feature            prob_feature(*index);
    doc_be_feature              be_feature(*index);
    doc_dph_feature             dph_feature(*index);
    doc_dfr_feature             dfr_feature(*index);
    doc_stream_feature          f_stream(*index);
    doc_tpscore_feature         f_tpscore(*index);

    for (auto &qry : qtfile.get_queries()) {
        std::vector<double>      stage0_scores = trec_run.get_scores(qry.id);
        std::vector<int>         docno_labels  = trec_run.get_labels(qry.id);
        std::vector<std::string> docnos        = trec_run.get_result(qry.id);
        std::vector<docid_t>     docids = qry_env.document_ids_from_metadata("docno", docnos);

        auto start = clock::now();

        for (size_t i = 0; i < docids.size(); ++i) {
            auto const docid = docids[i];
            auto const docno = docnos[i];
            auto const label = docno_labels[i];

            auto &freqs = fwd_idx[docid];
            freqs.q_ft  = calculate_q_freqs(*index, qry.stems);

            doc_entry             doc_entry(docid, freqs.pagerank);

            // set url_slash_count as feature for training
            doc_entry.url_slash_count = freqs.url_stats.url_slash_count;
            doc_entry.url_length      = freqs.url_stats.url_length;

            // set original run score as a feature for training
            doc_entry.stage0_score = stage0_scores[i];

            f_bm25_atire.compute(doc_entry, freqs);
            f_bm25_trec3.compute(doc_entry, freqs);
            f_bm25_trec3_kmax.compute(doc_entry, freqs);
            f_lmds_2500.compute(doc_entry, freqs);
            f_lmds_1500.compute(doc_entry, freqs);
            f_lmds_1000.compute(doc_entry, freqs);
            tfidf_feature.compute(doc_entry, freqs);
            prob_feature.compute(doc_entry, freqs);
            be_feature.compute(doc_entry, freqs);
            dph_feature.compute(doc_entry, freqs);
            dfr_feature.compute(doc_entry, freqs);
            f_stream.compute(doc_entry, freqs);
            features.compute(doc_entry, freqs);
            prox_feature.compute(doc_entry, qry);
            f_tpscore.compute(doc_entry, freqs);

            outfile << label << "," << qry.id << "," << docno << doc_entry << std::endl;
        }
        auto stop      = clock::now();
        auto load_time = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
        std::cerr << "qid: " << qry.id << ", "<< docids.size() << " docs in " << load_time.count() << " ms" << std::endl;
    }
    return 0;
}
