#include "cereal/archives/binary.hpp"
#include "cereal/types/map.hpp"
#include "indri/Repository.hpp"

#include <map>

using Lexicon = std::map<size_t, size_t>;

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

    Lexicon lexicon;

    indri::index::VocabularyIterator *iter = index->vocabularyIterator();
    iter->startIteration();

    size_t j = 0;
    while (!iter->finished()) {
        indri::index::DiskTermData *entry       = iter->currentEntry();
        indri::index::TermData *    termData    = entry->termData;
        uint32_t                    list_length = termData->corpus.documentCount;
        // dict_file << termData->term << " " << j << " " << termData->corpus.documentCount << " "
        //           << termData->corpus.totalCount << " " << std::endl;

        if (j % 1000000 == 0) {
            std::cerr << "Processing term " << j << ", " << termData->term << std::endl;
        }
        iter->nextEntry();
        j++;
    }
    delete iter;

    archive(lexicon);
    return 0;
}