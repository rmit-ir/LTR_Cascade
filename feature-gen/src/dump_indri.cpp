#include <iostream>
#include <unistd.h>

#include "indri/Repository.hpp"
#include "indri/CompressedCollection.hpp"

bool directory_exists(std::string dir) {
  struct stat sb;
  const char *pathname = dir.c_str();
  if (stat(pathname, &sb) == 0 && S_ISDIR(sb.st_mode)) {
    return true;
  }
  return false;
}

void create_directory(std::string dir) {
  if (!directory_exists(dir)) {
    if (mkdir(dir.c_str(), 0755) == -1) {
      perror("could not create directory");
      exit(EXIT_FAILURE);
    }
  }
}

int main(int argc, char **argv) {
  if (argc != 3) {
    std::cerr << "USAGE: " << argv[0]
              << " <indri repository> <newt collection folder>" << std::endl;
    return EXIT_FAILURE;
  }

  // parse cmd line
  std::string repository_name = argv[1];
  std::string newt_collection_folder = argv[2];
  create_directory(newt_collection_folder);
  std::string dict_file = newt_collection_folder + "/dict.txt";
  std::string doclen_file = newt_collection_folder + "/doc_lens.txt";
  std::string doc_names_file = newt_collection_folder + "/doc_names.txt";
  std::string int_file = newt_collection_folder + "/text.asc";
  std::string inv_file = newt_collection_folder + "/text.inv";
  std::string global_info_file = newt_collection_folder + "/global.txt";

  std::ofstream oifile(int_file);
  std::ofstream doclen_out(doclen_file);

  // load stuff
  indri::collection::Repository repo;
  repo.openRead(repository_name);
  indri::collection::Repository::index_state state = repo.indexes();
  const auto &index = (*state)[0];

  // dump global info; num documents in collection, num of all terms
  std::ofstream of_globalinfo(global_info_file);
  of_globalinfo << index->documentCount() << " " << index->termCount()
                << std::endl;

  // extract
  std::cerr << "Extracting documents from indri index." << std::endl;
  std::vector<std::string> document_names;

  uint64_t uniq_terms = index->uniqueTermCount();
  // Shift all ids from indri by 2 so \0 and \1 is free.
  uniq_terms += 2;
  indri::collection::CompressedCollection *collection = repo.collection();
  int64_t document_id = index->documentBase();
  indri::index::TermListFileIterator *iter = index->termListFileIterator();
  iter->startIteration();
  while (!iter->finished()) {
    indri::index::TermList *list = iter->currentEntry();

    // find document name
    std::string doc_name = collection->retrieveMetadatum(document_id, "docno");
    document_names.push_back(doc_name);

    if (document_id % 10000 == 0) {
      std::cout << ".";
      std::cout.flush();
    }

    doclen_out << list->terms().size() << std::endl;

    // iterate over termlist
    for (const auto &term_id : list->terms()) {
      // we will shift all ids from idri by 1 so \0 and \1 is free
      if (term_id != 0) {
        oifile << term_id + 1 << std::endl;
      }
    }
    oifile << 1 << std::endl;
    document_id++;
    iter->nextEntry();
  }

  // write document names
  {
    std::cerr << "Writing document names to " << doc_names_file << std::endl;
    std::ofstream of_doc_names(doc_names_file);
    for (const auto &doc_name : document_names) {
      of_doc_names << doc_name << std::endl;
    }
  }
  // write dictionary
  {
    std::cerr << "Writing dictionary to " << dict_file << std::endl;
    const auto &index = (*state)[0];
    std::ofstream of_dict(dict_file);

    indri::index::VocabularyIterator *iter = index->vocabularyIterator();
    iter->startIteration();
    std::cerr << "DIC TOTAL"
              << " " << index->termCount() << " " << index->documentCount()
              << std::endl;

    size_t i = 2;
    while (!iter->finished()) {
      indri::index::DiskTermData *entry = iter->currentEntry();
      indri::index::TermData *termData = entry->termData;

      of_dict << termData->term << " " << i << " "
              << termData->corpus.documentCount << " "
              << termData->corpus.totalCount << " " << std::endl;

      iter->nextEntry();
      i++;
    }
    delete iter;
  }

  // write inverted files
  {
    std::cerr << "Writing inverted file to " << inv_file << std::endl;
    const auto &index = (*state)[0];
    std::ofstream of_inv(inv_file);

    indri::index::DocListFileIterator *iter = index->docListFileIterator();
    iter->startIteration();

    while (!iter->finished()) {
      indri::index::DocListFileIterator::DocListData *entry =
          iter->currentEntry();
      indri::index::TermData *termData = entry->termData;

      entry->iterator->startIteration();

      of_inv << termData->term << " " << termData->corpus.totalCount << " "
             << termData->corpus.documentCount;

      while (!entry->iterator->finished()) {
        indri::index::DocListIterator::DocumentData *doc =
            entry->iterator->currentEntry();

        of_inv << " " << doc->document << ":" << doc->positions.size();
        entry->iterator->nextEntry();
      }
      of_inv << std::endl;
      iter->nextEntry();
    }
    delete iter;
  }
}
