#pragma once

class doc_stream_feature {
   public:

    void compute(query_train &qry, doc_entry &doc, FreqsEntry &freqs, FieldIdMap &field_id_map) {
        // stream length is set for the score member variables
        doc.stream_len       = freqs.doc_length;
        doc.stream_len_body  = freqs.field_len[field_id_map["body"]];
        doc.stream_len_title = freqs.field_len[field_id_map["title"]];
        // penalise docs with more than 1 title tag
        if (freqs.fields_stats.tags_count["title"] > 1) {
            doc.stream_len_title = -doc.stream_len_title;
        }
        doc.stream_len_heading = freqs.field_len[field_id_map["heading"]];
        doc.stream_len_inlink  = freqs.field_len[field_id_map["inlink"]];
        doc.stream_len_a       = freqs.field_len[field_id_map["a"]];

        double doc_tf     = 0;
        double body_tf    = 0;
        double title_tf   = 0;
        double heading_tf = 0;
        double inlink_tf  = 0;
        double a_tf       = 0;

        for (auto &q : qry.q_ft) {
            doc_tf += freqs.d_ft[q.first];
            body_tf += freqs.f_ft[{field_id_map["body"], q.first}];
            title_tf += freqs.f_ft[{field_id_map["title"], q.first}];
            heading_tf += freqs.f_ft[{field_id_map["heading"], q.first}];
            inlink_tf += freqs.f_ft[{field_id_map["inlink"], q.first}];
            a_tf += freqs.f_ft[{field_id_map["a"], q.first}];
        }

        if (doc_tf) {
            doc.sum_stream_len  = (double)freqs.doc_length / doc_tf;
            doc.min_stream_len  = doc.sum_stream_len;
            doc.max_stream_len  = doc.sum_stream_len;
            doc.mean_stream_len = doc.sum_stream_len;
            doc.variance_stream_len =
                ((double)freqs.doc_length - freqs.doc_length * freqs.doc_length) / doc_tf;
        }
        if (body_tf) {
            doc.sum_stream_len_body  = (double)freqs.field_len[field_id_map["body"]] / body_tf;
            doc.min_stream_len_body  = (double)freqs.field_min_len[field_id_map["body"]] / body_tf;
            doc.max_stream_len_body  = (double)freqs.field_max_len[field_id_map["body"]] / body_tf;
            doc.mean_stream_len_body = (double)((double)freqs.field_len[field_id_map["body"]] /
                                                freqs.fields_stats.tags_count["body"]) /
                                       body_tf;
            doc.variance_stream_len_body =
                (((double)freqs.field_len_sum_sqrs["body"] / freqs.field_len[field_id_map["body"]]) -
                 ((double)freqs.field_len[field_id_map["body"]] /
                  freqs.fields_stats.tags_count["body"]) *
                     ((double)freqs.field_len[field_id_map["body"]] /
                      freqs.fields_stats.tags_count["body"])) /
                body_tf;
        }
        if (title_tf) {
            doc.sum_stream_len_title = (double)freqs.field_len[field_id_map["title"]] / title_tf;
            doc.min_stream_len_title = (double)freqs.field_min_len[field_id_map["title"]] / title_tf;
            doc.max_stream_len_title = (double)freqs.field_max_len[field_id_map["title"]] / title_tf;
            doc.mean_stream_len_title = (double)((double)freqs.field_len[field_id_map["title"]] /
                                                 freqs.fields_stats.tags_count["title"]) /
                                        title_tf;
            doc.variance_stream_len_title = (((double)freqs.field_len_sum_sqrs["title"] /
                                              freqs.field_len[field_id_map["title"]]) -
                                             ((double)freqs.field_len[field_id_map["title"]] /
                                              freqs.fields_stats.tags_count["title"]) *
                                                 ((double)freqs.field_len[field_id_map["title"]] /
                                                  freqs.fields_stats.tags_count["title"])) /
                                            title_tf;
        }
        if (heading_tf) {
            doc.sum_stream_len_heading =
                (double)freqs.field_len[field_id_map["heading"]] / heading_tf;
            doc.min_stream_len_heading =
                (double)freqs.field_min_len[field_id_map["heading"]] / heading_tf;
            doc.max_stream_len_heading =
                (double)freqs.field_max_len[field_id_map["heading"]] / heading_tf;
            doc.mean_stream_len_heading = (double)((double)freqs.field_len[field_id_map["heading"]] /
                                                   freqs.fields_stats.tags_count["heading"]) /
                                          heading_tf;
            doc.variance_stream_len_heading =
                (((double)freqs.field_len_sum_sqrs["heading"] /
                  freqs.field_len[field_id_map["heading"]]) -
                 ((double)freqs.field_len[field_id_map["heading"]] /
                  freqs.fields_stats.tags_count["heading"]) *
                     ((double)freqs.field_len[field_id_map["heading"]] /
                      freqs.fields_stats.tags_count["heading"])) /
                heading_tf;
        }
        if (inlink_tf) {
            doc.sum_stream_len_inlink = (double)freqs.field_len[field_id_map["inlink"]] / inlink_tf;
            doc.min_stream_len_inlink =
                (double)freqs.field_min_len[field_id_map["inlink"]] / inlink_tf;
            doc.max_stream_len_inlink =
                (double)freqs.field_max_len[field_id_map["inlink"]] / inlink_tf;
            doc.mean_stream_len_inlink = (double)((double)freqs.field_len[field_id_map["inlink"]] /
                                                  freqs.fields_stats.tags_count["inlink"]) /
                                         inlink_tf;
            doc.variance_stream_len_inlink = (((double)freqs.field_len_sum_sqrs["inlink"] /
                                               freqs.field_len[field_id_map["inlink"]]) -
                                              ((double)freqs.field_len[field_id_map["inlink"]] /
                                               freqs.fields_stats.tags_count["inlink"]) *
                                                  ((double)freqs.field_len[field_id_map["inlink"]] /
                                                   freqs.fields_stats.tags_count["inlink"])) /
                                             inlink_tf;
        }
        if (a_tf) {
            doc.sum_stream_len_a  = (double)freqs.field_len[field_id_map["a"]] / a_tf;
            doc.min_stream_len_a  = (double)freqs.field_min_len[field_id_map["a"]] / a_tf;
            doc.max_stream_len_a  = (double)freqs.field_max_len[field_id_map["a"]] / a_tf;
            doc.mean_stream_len_a = (double)((double)freqs.field_len[field_id_map["a"]] /
                                             freqs.fields_stats.tags_count["a"]) /
                                    a_tf;
            doc.variance_stream_len_a =
                (((double)freqs.field_len_sum_sqrs["a"] / freqs.field_len[field_id_map["a"]]) -
                 ((double)freqs.field_len[field_id_map["a"]] / freqs.fields_stats.tags_count["a"]) *
                     ((double)freqs.field_len[field_id_map["a"]] /
                      freqs.fields_stats.tags_count["a"])) /
                a_tf;
        }
    }
};
