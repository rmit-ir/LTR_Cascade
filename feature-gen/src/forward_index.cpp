#include "indri/Repository.hpp"
#include "indri/CompressedCollection.hpp"
#include "indri/QueryEnvironment.hpp"

#include "freqs_entry.hpp"


size_t url_slash_count(const std::string& url) {
    size_t count = 0;
    const std::string proto = "://";
    const std::string param_delim = "?";
    size_t pos = url.find(proto);
    size_t pos_q = url.find(param_delim);

    if (pos_q < pos || std::string::npos == pos) {
      pos = 0;
    } else {
      pos += proto.size();
    }

    while (std::string::npos != (pos = url.find("/", pos + 1, 1))) {
      ++count;
    }

    return count;
}


static const std::vector<std::string> _fields = {"body", "title", "heading", "mainbody", "inlink", "a"};


int main(int argc, char const *argv[])
{
    if (argc != 3) {
    std::cerr << "USAGE: " << argv[0]
              << " <indri repository> <forward index name>" << std::endl;
    return EXIT_FAILURE;
    }

    const std::string repository_name = argv[1];
    const std::string forward_index = argv[2];
    std::ofstream os(forward_index, std::ios::binary);
    cereal::BinaryOutputArchive archive( os );

    indri::collection::Repository repo;
    repo.openRead(repository_name);
    indri::collection::Repository::index_state state = repo.indexes();
    const auto &index = (*state)[0];

    indri::api::QueryEnvironment indri_env;
    indri_env.addIndex(repository_name);

    FwdIdx fwd_idx;
    fwd_idx.push_back({});
    uint64_t docid = index->documentBase();
    indri::index::TermListFileIterator *iter = index->termListFileIterator();
    iter->startIteration();
    auto *priorIt =repo.priorListIterator("pagerank");
    priorIt->startIteration();
    while (!iter->finished()) {
        indri::index::TermList *list = iter->currentEntry();
        auto *priorEntry = priorIt->currentEntry();
        auto &doc_terms = list->terms();
        FreqsEntry freqs;
        auto url = indri_env.documentMetadata(std::vector<lemur::api::DOCID_T>{docid}, "url");
        freqs.url_stats.url_slash_count = url_slash_count(url.at(0));
        freqs.url_stats.url_length = url.at(0).size();

        for(auto& tid: doc_terms) {
          auto it = freqs.d_ft.find(tid);
          if (it == freqs.d_ft.end()) {
            freqs.d_ft[tid] = 1;
          }
          ++it->second;
        }

        freqs.doc_length = doc_terms.size();

        freqs.pagerank = priorEntry->score;

        auto fields = list->fields();
        for (auto &f : fields) {
            std::string field_name = index->field(f.id);
            ++freqs.fields_stats.tags_count[field_name];
        }

        for (const std::string &field_str : _fields) {
            int field_id = index->field(field_str);
            if (field_id < 1) {
              // field is not indexed
              continue;
            }
            freqs.field_max_len[field_id] = 0;
            freqs.field_min_len[field_id] = std::numeric_limits<int>::max();
            for (auto &f : fields) {
              if (f.id != static_cast<size_t>(field_id)) {
                continue;
              }

              freqs.field_len[field_id] += f.end - f.begin;


              if (freqs.field_max_len[field_id] < freqs.field_len[field_id]) {
                  freqs.field_max_len[field_id] = freqs.field_len[field_id];
              }

              if (freqs.field_min_len[field_id] > freqs.field_len[field_id]) {
                  freqs.field_min_len[field_id] = freqs.field_len[field_id];
              }

              freqs.field_len_sum_sqrs[field_str] += freqs.field_len[field_id] * freqs.field_len[field_id];

              // Possible BUG
              // Should I count for all the fields or for a single one and then aggregate?
              for (size_t i = f.begin; i < f.end; ++i) {
                // stc:: cerr << doc_terms[1] << " " << std::endl;
                auto field_term = std::make_pair(field_id, doc_terms[i]);
                auto it = freqs.f_ft.find(field_term);
                if (it == freqs.f_ft.end()) {
                  freqs.f_ft[field_term] = 1;
                } else {
                  ++it->second;
                }
              }
            }
        }
        fwd_idx.push_back(freqs);
        iter->nextEntry();
        priorIt->nextEntry();
        ++docid;
    }
    archive(fwd_idx);
    return 0;
}