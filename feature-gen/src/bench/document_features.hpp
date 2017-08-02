#ifndef DOCUMENT_FEATURES_HPP
#define DOCUMENT_FEATURES_HPP

#include <cstdint>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "fat.hpp"

#include "indri/Index.hpp"

namespace bench {

class document_features {
  using indri_index = indri::index::Index;

  indri_index &index;

  // The frequency of query terms within the <title> tag
  size_t tag_title_qry_count = 0;
  // The frequency of query terms within the <mainbody> tag
  size_t tag_mainbody_qry_count = 0;
  // The frequency of query terms within the h1-h4 tags (called heading by
  // Indri)
  size_t tag_heading_qry_count = 0;
  // The frequency of query terms within the inlinks
  size_t tag_inlink_qry_count = 0;

  // The number of times the <title> tag appears in the document
  int tag_title_count = 0;
  // The number of times a header tag (h1-h4) appears in the document
  int tag_heading_count = 0;
  // The number of inlinks in the document
  int tag_inlink_count = 0;
  // The number of times the <applet> tag appears in the document
  int tag_applet_count = 0;
  // The number of times the <object> tag appears in the document
  int tag_object_count = 0;
  // The number of times the <embed> tag appears in the document
  int tag_embed_count = 0;

  enum field_id : uint16_t {
    F_STR_NONE = 0,
    F_STR_TITLE,
    F_STR_HEADING,
    F_STR_MAINBODY,
    F_STR_INLINK,

    MARKER_STOP_QRY_COUNT,

    F_STR_APPLET,
    F_STR_OBJECT,
    F_STR_EMBED,
  };
  static std::map<std::string, uint16_t> field_lookup;

public:
  document_features(indri_index &i) : index(i) {}

  void compute(fat_cache_entry &doc, std::vector<std::string> &query_stems,
               std::string field_str = "") {
    tag_title_count = 0;
    tag_title_qry_count = 0;
    tag_heading_count = 0;
    tag_heading_qry_count = 0;
    tag_mainbody_qry_count = 0;
    tag_inlink_count = 0;
    tag_inlink_qry_count = 0;
    tag_applet_count = 0;
    tag_object_count = 0;
    tag_embed_count = 0;

    const indri::index::TermList *term_list = doc.term_list;

    /*
     * List of fields for the current document. The field `id` indicates which
     * field it is, e.g. title, heading, etc
     */
    auto field_vec = term_list->fields();

    std::map<uint64_t, uint32_t> counts;
    int field_id = index.field(field_str);
    if (field_id < 1) {
      // field does not exist
      return;
    }

    // reset query term counts
    for (auto &s : query_stems) {
      // a missing term has key `0`
      counts[index.term(s)] = 0;
    }

    auto &doc_terms = term_list->terms();
    for (auto &f : field_vec) {
      if (f.id != static_cast<size_t>(field_id)) {
        continue;
      }

      for (size_t i = f.begin; i < f.end; ++i) {
        auto it = counts.find(doc_terms[i]);
        if (it == counts.end()) {
          counts[doc_terms[i]] = 0;
        } else {
          ++it->second;
        }
      }
    }

    auto i = field_lookup.at(field_str);
    if (i > MARKER_STOP_QRY_COUNT) {
      // only count the number of tags
      return;
    }

    size_t qry_term_count = 0;
    for (auto el : counts) {
      qry_term_count += el.second;
    }
    set_tag_qry_count(field_str, qry_term_count);

    doc.tag_title_qry_count = tag_title_qry_count;
    doc.tag_heading_qry_count = tag_heading_qry_count;
    doc.tag_mainbody_qry_count = tag_mainbody_qry_count;
    doc.tag_inlink_qry_count = tag_inlink_qry_count;
  }

  void set_tag_count(std::string field, size_t n) {
    switch (field_lookup.at(field)) {
    case F_STR_TITLE:
      tag_title_count = n;
      if (tag_title_count > 1) {
        // penalise docs with more than 1 `title` tag
        tag_title_count = -tag_title_count;
      }
      break;

    case F_STR_HEADING:
      tag_heading_count = n;
      break;

    case F_STR_INLINK:
      tag_inlink_count = n;
      break;

    case F_STR_APPLET:
      tag_applet_count = n;
      break;

    case F_STR_OBJECT:
      tag_object_count = n;
      break;

    case F_STR_EMBED:
      tag_embed_count = n;
      break;

    case MARKER_STOP_QRY_COUNT:
    default:
      break;
    }
  }

  void set_tag_qry_count(std::string field, size_t n) {
    switch (field_lookup.at(field)) {
    case F_STR_TITLE:
      tag_title_qry_count = n;
      break;

    case F_STR_HEADING:
      tag_heading_qry_count = n;
      break;

    case F_STR_MAINBODY:
      tag_mainbody_qry_count = n;
      break;

    case F_STR_INLINK:
      tag_inlink_qry_count = n;
      break;

    case MARKER_STOP_QRY_COUNT:
    default:
      break;
    }
  }
};

std::map<std::string, uint16_t> document_features::field_lookup = {
    {"title", F_STR_TITLE},       {"heading", F_STR_HEADING},
    {"mainbody", F_STR_MAINBODY}, {"inlink", F_STR_INLINK},
    {"applet", F_STR_APPLET},     {"object", F_STR_OBJECT},
    {"embed", F_STR_EMBED},
};

} /* bench */

#endif
