#include "cereal/archives/binary.hpp"
#include "indri/Repository.hpp"

#include "lexicon.hpp"

static const std::vector<std::string> _fields = {
    "body", "title", "heading", "mainbody", "inlink", "a"};

int main(int argc, char const *argv[]) {
    if (argc != 3) {
        std::cerr << "USAGE: " << argv[0] << " <indri repository> <lexicon name>" << std::endl;
        return EXIT_FAILURE;
    }

    const std::string repository_name  = argv[1];
    const std::string lexicon_filename = argv[2];

    std::ofstream               os(lexicon_filename, std::ios::binary);
    cereal::BinaryOutputArchive archive(os);

    indri::collection::Repository repo;
    repo.openRead(repository_name);
    indri::collection::Repository::index_state state = repo.indexes();
    const auto &                               index = (*state)[0];

    indri::index::VocabularyIterator *iter = index->vocabularyIterator();
    iter->startIteration();

    Lexicon lexicon(Counts(index->documentCount(), index->termCount()));
    lexicon.push_back({});

    while (!iter->finished()) {
        indri::index::DiskTermData *entry    = iter->currentEntry();
        indri::index::TermData *    termData = entry->termData;


        FieldCounts field_counts;
        for (const std::string &field_str : _fields) {
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