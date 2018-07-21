#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

#include "cereal/archives/binary.hpp"
#include "doc_entry.hpp"
#include "field_id.hpp"

#include "features/features.hpp"
#include "freqs_entry.hpp"

#include "lexicon.hpp"

#include "query_train_file.hpp"
#include "trec_run_file.hpp"

static void display_usage(char *name) {
    std::cerr << "usage: " << name
              << " <query file> <trec file> <indri repository> <forward index> <lexicon> "
                 "<output file>"
              << std::endl;
    exit(EXIT_FAILURE);
}

/**
 * Generate document features and output to CSV format.
 */
int main(int argc, char **argv) {
    if (argc < 7) {
        display_usage(argv[0]);
    }

    std::string query_file         = argv[1];
    std::string trec_file          = argv[2];
    std::string repo_path          = argv[3];
    std::string forward_index_file = argv[4];
    std::string lexicon_file       = argv[5];

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

    // load trec run file
    ifs.open(trec_file);
    trec_run_file trec_run(ifs);
    trec_run.parse();
    ifs.close();
    ifs.clear();

    using clock = std::chrono::high_resolution_clock;
    auto start  = clock::now();

    // load fwd_idx
    std::ifstream              ifs_fwd(forward_index_file);
    cereal::BinaryInputArchive iarchive_fwd(ifs_fwd);
    FwdIdx                     fwd_idx;
    iarchive_fwd(fwd_idx);

    auto stop      = clock::now();
    auto load_time = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    std::cerr << "Loaded " << forward_index_file << " in " << load_time.count() << " ms"
              << std::endl;

    start  = clock::now();
    // load lexicon
    std::ifstream              lexicon_f(lexicon_file);
    cereal::BinaryInputArchive iarchive_lex(lexicon_f);
    Lexicon                    lexicon;
    iarchive_lex(lexicon);

    stop      = clock::now();
    load_time = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    std::cerr << "Loaded " << lexicon_file << " in " << load_time.count() << " ms"
              << std::endl;

    FieldIdMap field_id_map;
    const std::vector<std::string> idx_fields = {
        "title", "heading", "mainbody", "inlink", "applet", "object", "embed"};
    for (const std::string &field_str : idx_fields) {
        int field_id = index->field(field_str);
        field_id_map.insert(std::make_pair(field_str, field_id));
    }

    document_features           features;
    doc_bm25_atire_feature      f_bm25_atire(lexicon);
    doc_bm25_trec3_feature      f_bm25_trec3(lexicon);
    doc_bm25_trec3_kmax_feature f_bm25_trec3_kmax(lexicon);
    doc_proximity_feature       prox_feature(lexicon);
    doc_lm_dir_2500_feature     f_lmds_2500(lexicon);
    doc_lm_dir_1500_feature     f_lmds_1500(lexicon);
    doc_lm_dir_1000_feature     f_lmds_1000(lexicon);
    doc_tfidf_feature           tfidf_feature(lexicon);
    doc_prob_feature            prob_feature(lexicon);
    doc_be_feature              be_feature(lexicon);
    doc_dph_feature             dph_feature(lexicon);
    doc_dfr_feature             dfr_feature(lexicon);
    doc_stream_feature          f_stream;
    doc_tpscore_feature         f_tpscore(lexicon);

    auto queries = qtfile.get_queries();
    for (auto &qry : queries) {
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

            doc_entry doc_entry(docid, freqs.pagerank);

            // set url_slash_count as feature for training
            doc_entry.url_slash_count = freqs.url_stats.url_slash_count;
            doc_entry.url_length      = freqs.url_stats.url_length;

            // set original run score as a feature for training
            doc_entry.stage0_score = stage0_scores[i];

            f_bm25_atire.compute(doc_entry, freqs, field_id_map);
            f_bm25_trec3.compute(doc_entry, freqs, field_id_map);
            f_bm25_trec3_kmax.compute(doc_entry, freqs, field_id_map);
            f_lmds_2500.compute(doc_entry, freqs, field_id_map);
            f_lmds_1500.compute(doc_entry, freqs, field_id_map);
            f_lmds_1000.compute(doc_entry, freqs, field_id_map);
            tfidf_feature.compute(doc_entry, freqs, field_id_map);
            prob_feature.compute(doc_entry, freqs, field_id_map);
            be_feature.compute(doc_entry, freqs, field_id_map);
            dph_feature.compute(doc_entry, freqs, field_id_map);
            dfr_feature.compute(doc_entry, freqs, field_id_map);
            f_stream.compute(doc_entry, freqs, field_id_map);
            features.compute(doc_entry, freqs, field_id_map);
            prox_feature.compute(doc_entry, qry, freqs);
            f_tpscore.compute(doc_entry, freqs, field_id_map);

            outfile << label << "," << qry.id << "," << docno << doc_entry << std::endl;
        }
        auto stop      = clock::now();
        auto load_time = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
        std::cerr << "qid: " << qry.id << ", " << docids.size() << " docs in " << load_time.count()
                  << " ms" << std::endl;
    }
    return 0;
}
