#include "fgen_bigram.hpp"

int main(int argc, char **argv) {
    FILE *   in_file = NULL;
    uint64_t cf      = 0;
    char     term_a[4096];
    char     term_b[4096];
    uint64_t pos   = 0;
    uint32_t docid = 0;
    uint32_t df = 0, i = 0;
    int      done      = 0;
    int      freq      = 0;
    double   tfidf_max = 0.0;
    double   bm25_max  = 0.0;
    double   pr_max    = 0.0;
    double   be_max    = 0.0;
    double   dfr_max   = 0.0;
    double   dph_max   = 0.0;
    double   lm_max    = -DBL_MAX;
    /* ONLY for CLUEWEB09B! */
    // uint64_t clen = 40541601698;
    uint64_t   clen     = 0;
    cmd_opt_t *cmd_line = NULL;

    int       ndocs    = 0;
    uint32_t *doclen   = NULL;
    double    avg_dlen = 0.0;

    cmd_line = parse_args(argc, argv);

    clen = cmd_line->collen;
    fprintf(stderr, "Collection Length <%" SCNu64 ">\n", clen);

    ndocs = 0;
    fprintf(stderr, "Loading doclen from %s\n", cmd_line->dlen_file_name);
    doclen = build_doclen(cmd_line->dlen_file_name, &ndocs, &avg_dlen);
    fprintf(stderr, "Avg Document Length: %f\n", avg_dlen);

    // printf ("Opening -- %s\n", cmd_line->idx_file_name);
    if ((in_file = fopen(cmd_line->idx_file_name, "rb")) == NULL) {
        fprintf(stderr, "fopen(%s)\n", cmd_line->idx_file_name);
        exit(EXIT_FAILURE);
    }

    /* Walk inverted file dump from Indri. */

    // printf ("BEGIN\n");
    while (fscanf(in_file, "%s %s", term_a, term_b) == 2) {
        feature_t *f    = (feature_t *)safe_malloc(sizeof(feature_t));
        int        sz_a = 0, sz_b = 0;
        fscanf(in_file, "%" SCNu64, &cf);
        fscanf(in_file, "%" SCNu64, &pos);

        f->cf  = cf;
        f->cdf = pos;
        sz_a   = strlen(term_a);
        sz_b   = strlen(term_b);
        if (sz_a < 1023 && sz_b < 1023) {
            // printf ("-");
            strcpy(f->term_a, term_a);
            strcpy(f->term_b, term_b);
            // printf ("--");
        } else {
            fprintf(stderr, "ERROR term_a <%s> too long <%d>.", term_a, sz_a);
            fprintf(stderr, "ERROR term_b <%s> too long <%d>.", term_b, sz_b);
            exit(EXIT_FAILURE);
        }

        /* Min count is set to 4 or IQR computation goes boom. */
        if (pos >= 4) {
            posting_t *posting = (posting_t *)safe_malloc(sizeof(posting_t) * pos);
            for (i = 0; i < pos; i++) {
                fscanf(in_file, "%" SCNu32 ":", &docid);
                fscanf(in_file, "%" SCNu32, &df);
                posting[i].docid = docid;
                posting[i].freq  = df;
                // score = calculate_bm25 (doclen[docid-1], avg_dlen, ndocs, pos, df);
                // score = calculate_tfidf (df, pos, doclen[docid-1], ndocs);
            }
            f->geo_mean = compute_geo_mean(posting, pos);
            compute_tfidf_stats(f, doclen, posting, pos, ndocs, &tfidf_max);
            compute_bm25_stats(f, doclen, posting, pos, ndocs, avg_dlen, &bm25_max);
            compute_lm_stats(f, doclen, posting, pos, clen, cf, &lm_max);
            compute_prob_stats(f, doclen, posting, pos, &pr_max);
            compute_be_stats(f, doclen, posting, pos, ndocs, avg_dlen, cf, &be_max);
            compute_dph_stats(f, doclen, posting, pos, ndocs, avg_dlen, cf, &dph_max);
            compute_dfr_stats(f, doclen, posting, pos, ndocs, avg_dlen, cf, &dfr_max);
            free(posting);
            print_feature(f);
            freq++;
        } else {
            for (i = 0; i < pos; i++) {
                /* Read but ignore really short lists */
                fscanf(in_file, "%" SCNu32 ":", &docid);
                fscanf(in_file, "%" SCNu32, &df);
            }
        }
        done++;
        free(f);
    }

    fprintf(stderr, "Inv Lists Processed = %d\n", done);
    fprintf(stderr, "Inv Lists > 4 = %d\n", freq);
    fprintf(stderr, "TFIDF Max Score = %lf\n", tfidf_max);
    fprintf(stderr, "BM25 Max Score = %lf\n", bm25_max);
    fprintf(stderr, "LM Max Score = %lf\n", lm_max);
    fprintf(stderr, "PR Max Score = %lf\n", pr_max);
    fprintf(stderr, "BE Max Score = %lf\n", be_max);
    fprintf(stderr, "DPH Max Score = %lf\n", dph_max);
    fprintf(stderr, "DFR Max Score = %lf\n", dfr_max);
    fclose(in_file);
    free(doclen);
    destroy_cmd_opt_t(cmd_line);
    return (EXIT_SUCCESS);
}
