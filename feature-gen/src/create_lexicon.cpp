#include "cereal/archives/binary.hpp"
#include "CLI/CLI.hpp"

#include "indri/Repository.hpp"
#include "indri/QueryEnvironment.hpp"

#include "lexicon.hpp"

int main(int argc, char const *argv[]) {
    std::string repo_path;
    std::string lexicon_file;

    CLI::App app{"Lexicon generator."};
    app.add_option("repo_path", repo_path, "Indri repo path")->required();
    app.add_option("lexicon_file", lexicon_file, "Lexicon file")->required();
    CLI11_PARSE(app, argc, argv);


    std::ofstream               os(lexicon_file, std::ios::binary);
    cereal::BinaryOutputArchive archive(os);

    indri::collection::Repository repo;
    repo.openRead(repo_path);
    indri::collection::Repository::index_state state = repo.indexes();
    const auto &                               index = (*state)[0];

    indri::api::QueryEnvironment         env;
    env.addIndex(repo_path);
    auto fields = env.fieldList();

    indri::index::VocabularyIterator *iter = index->vocabularyIterator();
    iter->startIteration();

    Lexicon lexicon(Counts(index->documentCount(), index->termCount()));
    lexicon.push_back({});

    while (!iter->finished()) {
        indri::index::DiskTermData *entry    = iter->currentEntry();
        indri::index::TermData *    termData = entry->termData;


        FieldCounts field_counts;
        for (const std::string &field_str : fields) {
            int field_id = index-> field(field_str);
            Counts c(termData->fields[field_id - 1].documentCount, termData->fields[field_id - 1].totalCount);
            field_counts.insert(std::make_pair(field_id, c));
        }
        Counts counts(termData->corpus.documentCount, termData->corpus.totalCount);
        lexicon.push_back(termData->term, counts, field_counts);
        iter->nextEntry();
    }
    delete iter;

    archive(lexicon);
    return 0;
}