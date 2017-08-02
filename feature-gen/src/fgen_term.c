#include <assert.h>
#include <ctype.h>
#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <inttypes.h>
#include <float.h>
#include <getopt.h>

#include <sys/time.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>

#ifndef TRUE
#define TRUE 1
#endif
#ifndef FALSE
#define FALSE 0
#endif

typedef struct {
  char *idx_file_name;
  char *dlen_file_name;
  uint64_t collen;
  int verbose;
} cmd_opt_t;

typedef struct {
  char term[1024];
  uint64_t cf;
  uint64_t cdf;
  double geo_mean;
  double tfidf_median;
  double tfidf_first;
  double tfidf_third;
  double tfidf_min;
  double tfidf_max;
  double tfidf_avg;
  double tfidf_variance;
  double tfidf_std_dev;
  double tfidf_confidence;
  double tfidf_hmean;
  double bm25_median;
  double bm25_first;
  double bm25_third;
  double bm25_min;
  double bm25_max;
  double bm25_avg;
  double bm25_variance;
  double bm25_std_dev;
  double bm25_confidence;
  double bm25_hmean;
  double lm_median;
  double lm_first;
  double lm_third;
  double lm_min;
  double lm_max;
  double lm_avg;
  double lm_variance;
  double lm_std_dev;
  double lm_confidence;
  double lm_hmean;
  double dfr_median;
  double dfr_first;
  double dfr_third;
  double dfr_min;
  double dfr_max;
  double dfr_avg;
  double dfr_variance;
  double dfr_std_dev;
  double dfr_confidence;
  double dfr_hmean;
  double dph_median;
  double dph_first;
  double dph_third;
  double dph_min;
  double dph_max;
  double dph_avg;
  double dph_variance;
  double dph_std_dev;
  double dph_confidence;
  double dph_hmean;
  double be_median;
  double be_first;
  double be_third;
  double be_min;
  double be_max;
  double be_avg;
  double be_variance;
  double be_std_dev;
  double be_confidence;
  double be_hmean;
  double pr_median;
  double pr_first;
  double pr_third;
  double pr_min;
  double pr_max;
  double pr_avg;
  double pr_variance;
  double pr_std_dev;
  double pr_confidence;
  double pr_hmean;
} feature_t;

typedef struct {
  uint64_t docid;
  uint32_t freq;
} posting_t;

void *safe_malloc(size_t size) {
  void *mem_block = NULL;

  if ((mem_block = calloc(1, size)) == NULL) {
    fprintf(stderr, "ERROR: safe_malloc(%lu) cannot allocate memory.", size);
    exit(EXIT_FAILURE);
  }
  return (mem_block);
}

void *safe_realloc(void *old_mem, size_t new_size) {
  if ((old_mem = realloc(old_mem, new_size)) == NULL) {
    fprintf(stderr, "ERROR: safe_realloc() cannot allocate"
                    "%u blocks of memory.\n",
            (unsigned int)new_size);
    exit(EXIT_FAILURE);
  }
  return (old_mem);
}

char *safe_strdup(const char *str) {
  char *copy = NULL;

  if (str == NULL) {
    fprintf(stderr, "ERROR safe_strdup(): str == NULL");
    exit(EXIT_FAILURE);
  }

  copy = safe_malloc((strlen(str) + 1) * sizeof(char));

  (void)strcpy(copy, str);

  return (copy);
}

uint32_t *build_doclen(char *dfile, int *ndocs, double *avg_dlen) {
  int dmsize = 4096;
  uint32_t len = 0;
  FILE *docfile = NULL;
  uint32_t *dlen = NULL;
  uint64_t dl_sum = 0;

  dlen = (uint32_t *)safe_malloc(4096 * sizeof(uint32_t));

  if ((docfile = fopen(dfile, "rb")) == NULL) {
    fprintf(stderr, "ERROR: fopen(%s)\n", dfile);
    exit(EXIT_FAILURE);
  }

  while (fscanf(docfile, "%" SCNu32 ":", &len) == 1) {
    dl_sum += len;
    if (dmsize == *ndocs) {
      uint32_t *tmpa =
          (uint32_t *)safe_realloc(dlen, dmsize * 2 * sizeof(uint32_t));
      dlen = tmpa;
      dmsize *= 2;
    }
    dlen[*ndocs] = len;
    *ndocs += 1;
  }
  /* Freeze doclen to final size */
  uint32_t *tmpa = (uint32_t *)safe_realloc(dlen, (*ndocs) * sizeof(uint32_t));
  dlen = tmpa;
  fclose(docfile);
  *avg_dlen = (double)dl_sum / *ndocs;
  return (dlen);
}

double calculate_bm25(uint32_t dlen, double avg_dlen, uint32_t ndocs,
                      uint32_t t_idf, uint32_t d_f)

{
  double k1 = 0.9, b = 0.4;
  double score = 0.0;

  double tmp1 = 0.0, tmp2 = 0.0, idf = 0.0;
  double d_l = dlen;
  double f_dt = d_f;
  double f_t = t_idf;
  tmp1 = f_dt + k1 * ((1 - b) + (b * d_l / avg_dlen));
  tmp2 = ((k1 + 1) * f_dt) / tmp1;
  idf = log((double)(ndocs - f_t + 0.5) / (double)(f_t + 0.5));
  if (idf < 0) {
    idf = 0;
  }
  score = tmp2 * idf;
  return (score);
}

double calculate_tfidf(double d_f, double t_idf, double dlen,
                       uint32_t num_docs) {
  double doc_norm = 1.0 / dlen;
  double w_dq = 1.0 + log(d_f);
  double w_Qq = log(1.0 + ((double)num_docs / t_idf));
  return (doc_norm * w_dq * w_Qq);
}

double calculate_lm(uint32_t d_f, uint64_t c_f, uint32_t dlen, uint64_t clen) {

  double mu = 2500.00;
  double numerator = d_f + mu * c_f / clen;
  double denominator = dlen + mu;
  return (log(numerator / denominator));
}

double calculate_prob(uint32_t d_f, uint32_t dlen) {
  return ((double)d_f / dlen);
}

double calculate_bose_ein(uint32_t d_f, uint64_t c_f, uint32_t num_docs,
                          double avg_dlen, uint32_t dlen) {
  double l, r, prime, rsv;
  l = log(1.0 + (double)c_f / num_docs);
  r = log(1.0 + (double)num_docs / (double)c_f);
  prime = d_f * log(1.0 + avg_dlen / (double)dlen);
  rsv = (l + prime * r) / (prime + 1.0);
  return (rsv);
}

double calculate_dph(uint32_t d_f, uint64_t c_f, uint32_t num_docs,
                     double avg_dlen, uint32_t dlen) {
  double f, norm, score;
  f = (double)d_f / (double)dlen;
  norm = (1.0 - f) * (1.0 - f) / (d_f + 1.0);
  score = 1.0 * norm *
          ((double)d_f * log2(((double)d_f * (double)avg_dlen / (double)dlen) *
                              ((double)num_docs / (double)c_f)) +
           0.5 * log2(2.0 * M_PI * d_f * (1.0 - f)));
  return (score);
}

double calculate_dfr(uint32_t d_f, uint64_t c_f, uint32_t c_idf,
                     uint32_t num_docs, double avg_dlen, uint32_t dlen) {
  double fp1, ne, ir, prime, rsv;

  fp1 = c_f + 1.0;
  ne = num_docs * (1.0 - pow((num_docs - 1.0) / num_docs, c_f));
  ir = log2(((double)num_docs + 1.0) / (ne + 0.5));

  prime = d_f * log2(1.0 + (double)avg_dlen / (double)dlen);
  rsv = prime * ir * (fp1 / ((double)c_idf * (prime + 1.0)));
  return (rsv);
}

int compare(const void *a, const void *b) {
  double fa = *(const double *)a;
  double fb = *(const double *)b;
  return (fa < fb) - (fa > fb);
}

double compute_geo_mean(posting_t *posting, uint64_t pos) {
  uint64_t i;
  double sum = 0.0;
  double geo_mean = 0.0;

  for (i = 0; i < pos; i++) {
    sum += posting[i].freq;
  }

  geo_mean = pow(sum, (1.0 / pos));
  return (geo_mean);
}

void compute_prob_stats(feature_t *f, uint32_t *doclen, posting_t *posting,
                        uint64_t pos, double *max) {
  uint64_t i;
  uint32_t size = pos;
  uint32_t mid = size / 2;
  uint32_t lq = size / 4;
  uint32_t uq = 3 * size / 4;
  double sum = 0.0, sum_sqrs = 0.0, variance = 0.0;
  double hmsum = 0.0;
  double zeta = 1.960;
  double *bmtmp = (double *)safe_malloc(sizeof(double) * pos);

  for (i = 0; i < pos; i++) {
    double score = 0.0;
    score = calculate_prob(posting[i].freq, doclen[posting[i].docid - 1]);
    bmtmp[i] = score;
    if (score > *max)
      *max = score;
  }

  qsort(bmtmp, pos, sizeof(double), compare);

  f->pr_median = size % 2 == 0 ? (bmtmp[mid] + bmtmp[mid - 1]) / 2 : bmtmp[mid];
  f->pr_first = size % 2 == 0 ? (bmtmp[lq] + bmtmp[lq - 1]) / 2 : bmtmp[lq];
  f->pr_third = size % 2 == 0 ? (bmtmp[uq] + bmtmp[uq - 1]) / 2 : bmtmp[uq];
  f->pr_max = bmtmp[0];
  f->pr_min = bmtmp[pos - 1];
  for (i = 0; i < pos; i++) {
    sum += bmtmp[i];
    sum_sqrs += bmtmp[i] * bmtmp[i];
    hmsum += 1 / bmtmp[i];
  }

  f->pr_avg = sum / (double)pos;
  f->pr_variance = (sum_sqrs / (double)pos) - f->pr_avg * f->pr_avg;
  f->pr_std_dev = sqrt(variance);
  f->pr_confidence = zeta * (f->pr_std_dev / (sqrt(pos)));
  f->pr_hmean = (double)pos / hmsum;

  free(bmtmp);
  return;
}

void compute_be_stats(feature_t *f, uint32_t *doclen, posting_t *posting,
                      uint64_t pos, uint64_t ndocs, double avg_dlen,
                      uint64_t c_f, double *max) {
  uint64_t i;
  uint32_t size = pos;
  uint32_t mid = size / 2;
  uint32_t lq = size / 4;
  uint32_t uq = 3 * size / 4;
  double sum = 0.0, sum_sqrs = 0.0, variance = 0.0;
  double hmsum = 0.0;
  double zeta = 1.960;
  double *bmtmp = (double *)safe_malloc(sizeof(double) * pos);

  for (i = 0; i < pos; i++) {
    double score = 0.0;
    score = calculate_bose_ein(posting[i].freq, c_f, ndocs, avg_dlen,
                               doclen[posting[i].docid - 1]);
    bmtmp[i] = score;
    if (score > *max)
      *max = score;
  }

  qsort(bmtmp, pos, sizeof(double), compare);

  f->be_median = size % 2 == 0 ? (bmtmp[mid] + bmtmp[mid - 1]) / 2 : bmtmp[mid];
  f->be_first = size % 2 == 0 ? (bmtmp[lq] + bmtmp[lq - 1]) / 2 : bmtmp[lq];
  f->be_third = size % 2 == 0 ? (bmtmp[uq] + bmtmp[uq - 1]) / 2 : bmtmp[uq];
  f->be_max = bmtmp[0];
  f->be_min = bmtmp[pos - 1];
  for (i = 0; i < pos; i++) {
    sum += bmtmp[i];
    sum_sqrs += bmtmp[i] * bmtmp[i];
    hmsum += 1 / bmtmp[i];
  }

  f->be_avg = sum / (double)pos;
  f->be_variance = (sum_sqrs / (double)pos) - f->be_avg * f->be_avg;
  f->be_std_dev = sqrt(variance);
  f->be_confidence = zeta * (f->be_std_dev / (sqrt(pos)));
  f->be_hmean = (double)pos / hmsum;

  free(bmtmp);
  return;
}

void compute_dph_stats(feature_t *f, uint32_t *doclen, posting_t *posting,
                       uint64_t pos, uint64_t ndocs, double avg_dlen,
                       uint64_t c_f, double *max) {
  uint64_t i;
  uint32_t size = pos;
  uint32_t mid = size / 2;
  uint32_t lq = size / 4;
  uint32_t uq = 3 * size / 4;
  double sum = 0.0, sum_sqrs = 0.0, variance = 0.0;
  double hmsum = 0.0;
  double zeta = 1.960;
  double *bmtmp = (double *)safe_malloc(sizeof(double) * pos);

  for (i = 0; i < pos; i++) {
    double score = 0.0;
    score = calculate_dph(posting[i].freq, c_f, ndocs, avg_dlen,
                          doclen[posting[i].docid - 1]);
    bmtmp[i] = score;
    if (score > *max)
      *max = score;
  }

  qsort(bmtmp, pos, sizeof(double), compare);

  f->dph_median =
      size % 2 == 0 ? (bmtmp[mid] + bmtmp[mid - 1]) / 2 : bmtmp[mid];
  f->dph_first = size % 2 == 0 ? (bmtmp[lq] + bmtmp[lq - 1]) / 2 : bmtmp[lq];
  f->dph_third = size % 2 == 0 ? (bmtmp[uq] + bmtmp[uq - 1]) / 2 : bmtmp[uq];
  f->dph_max = bmtmp[0];
  f->dph_min = bmtmp[pos - 1];
  for (i = 0; i < pos; i++) {
    sum += bmtmp[i];
    sum_sqrs += bmtmp[i] * bmtmp[i];
    hmsum += 1 / bmtmp[i];
  }

  f->dph_avg = sum / (double)pos;
  f->dph_variance = (sum_sqrs / (double)pos) - f->dph_avg * f->dph_avg;
  f->dph_std_dev = sqrt(variance);
  f->dph_confidence = zeta * (f->dph_std_dev / (sqrt(pos)));
  f->dph_hmean = (double)pos / hmsum;

  free(bmtmp);
  return;
}

void compute_dfr_stats(feature_t *f, uint32_t *doclen, posting_t *posting,
                       uint64_t pos, uint64_t ndocs, double avg_dlen,
                       uint64_t c_f, double *max) {
  uint64_t i;
  uint32_t size = pos;
  uint32_t mid = size / 2;
  uint32_t lq = size / 4;
  uint32_t uq = 3 * size / 4;
  double sum = 0.0, sum_sqrs = 0.0, variance = 0.0;
  double hmsum = 0.0;
  double zeta = 1.960;
  double *bmtmp = (double *)safe_malloc(sizeof(double) * pos);

  for (i = 0; i < pos; i++) {
    double score = 0.0;
    score = calculate_dfr(posting[i].freq, c_f, pos, ndocs, avg_dlen,
                          doclen[posting[i].docid - 1]);
    bmtmp[i] = score;
    if (score > *max)
      *max = score;
  }

  qsort(bmtmp, pos, sizeof(double), compare);

  f->dfr_median =
      size % 2 == 0 ? (bmtmp[mid] + bmtmp[mid - 1]) / 2 : bmtmp[mid];
  f->dfr_first = size % 2 == 0 ? (bmtmp[lq] + bmtmp[lq - 1]) / 2 : bmtmp[lq];
  f->dfr_third = size % 2 == 0 ? (bmtmp[uq] + bmtmp[uq - 1]) / 2 : bmtmp[uq];
  f->dfr_max = bmtmp[0];
  f->dfr_min = bmtmp[pos - 1];
  for (i = 0; i < pos; i++) {
    sum += bmtmp[i];
    sum_sqrs += bmtmp[i] * bmtmp[i];
    hmsum += 1 / bmtmp[i];
  }

  f->dfr_avg = sum / (double)pos;
  f->dfr_variance = (sum_sqrs / (double)pos) - f->dfr_avg * f->dfr_avg;
  f->dfr_std_dev = sqrt(variance);
  f->dfr_confidence = zeta * (f->dfr_std_dev / (sqrt(pos)));
  f->dfr_hmean = (double)pos / hmsum;

  free(bmtmp);
  return;
}

void compute_tfidf_stats(feature_t *f, uint32_t *doclen, posting_t *posting,
                         uint64_t pos, uint64_t ndocs, double *max) {
  uint64_t i;
  uint32_t size = pos;
  uint32_t mid = size / 2;
  uint32_t lq = size / 4;
  uint32_t uq = 3 * size / 4;
  double sum = 0.0, sum_sqrs = 0.0, variance = 0.0;
  double hmsum = 0.0;
  double zeta = 1.960;
  double *bmtmp = (double *)safe_malloc(sizeof(double) * pos);

  for (i = 0; i < pos; i++) {
    double score = 0.0;
    // fscanf (in_file, "%" SCNu32 ":", &docid);
    // fscanf (in_file, "%" SCNu32, &df);
    // score = calculate_bm25 (doclen[docid-1], avg_dlen, ndocs, pos, df);
    // score = calculate_lm (df, cf, doclen[docid-1], clen);
    score = calculate_tfidf(posting[i].freq, pos, doclen[posting[i].docid - 1],
                            ndocs);
    bmtmp[i] = score;
    if (score > *max)
      *max = score;
  }

  qsort(bmtmp, pos, sizeof(double), compare);

  f->tfidf_median =
      size % 2 == 0 ? (bmtmp[mid] + bmtmp[mid - 1]) / 2 : bmtmp[mid];
  f->tfidf_first = size % 2 == 0 ? (bmtmp[lq] + bmtmp[lq - 1]) / 2 : bmtmp[lq];
  f->tfidf_third = size % 2 == 0 ? (bmtmp[uq] + bmtmp[uq - 1]) / 2 : bmtmp[uq];
  f->tfidf_max = bmtmp[0];
  f->tfidf_min = bmtmp[pos - 1];
  for (i = 0; i < pos; i++) {
    sum += bmtmp[i];
    sum_sqrs += bmtmp[i] * bmtmp[i];
    hmsum += 1 / bmtmp[i];
  }

  f->tfidf_avg = sum / (double)pos;
  f->tfidf_variance = (sum_sqrs / (double)pos) - f->tfidf_avg * f->tfidf_avg;
  f->tfidf_std_dev = sqrt(variance);
  f->tfidf_confidence = zeta * (f->tfidf_std_dev / (sqrt(pos)));
  f->tfidf_hmean = (double)pos / hmsum;

  free(bmtmp);
  return;
}

void compute_bm25_stats(feature_t *f, uint32_t *doclen, posting_t *posting,
                        uint64_t pos, uint64_t ndocs, double avg_dlen,
                        double *max) {
  uint64_t i;
  uint32_t size = pos;
  uint32_t mid = size / 2;
  uint32_t lq = size / 4;
  uint32_t uq = 3 * size / 4;
  double sum = 0.0, sum_sqrs = 0.0, variance = 0.0;
  double hmsum = 0.0;
  double zeta = 1.960;
  double *bmtmp = (double *)safe_malloc(sizeof(double) * pos);

  for (i = 0; i < pos; i++) {
    double score = 0.0;
    score = calculate_bm25(doclen[posting[i].docid - 1], avg_dlen, ndocs, pos,
                           posting[i].freq);
    bmtmp[i] = score;
    if (score > *max)
      *max = score;
  }

  qsort(bmtmp, pos, sizeof(double), compare);

  f->bm25_median =
      size % 2 == 0 ? (bmtmp[mid] + bmtmp[mid - 1]) / 2 : bmtmp[mid];
  f->bm25_first = size % 2 == 0 ? (bmtmp[lq] + bmtmp[lq - 1]) / 2 : bmtmp[lq];
  f->bm25_third = size % 2 == 0 ? (bmtmp[uq] + bmtmp[uq - 1]) / 2 : bmtmp[uq];
  f->bm25_max = bmtmp[0];
  f->bm25_min = bmtmp[pos - 1];
  for (i = 0; i < pos; i++) {
    sum += bmtmp[i];
    sum_sqrs += bmtmp[i] * bmtmp[i];
    hmsum += 1 / bmtmp[i];
  }

  f->bm25_avg = sum / (double)pos;
  f->bm25_variance = (sum_sqrs / (double)pos) - f->bm25_avg * f->bm25_avg;
  f->bm25_std_dev = sqrt(variance);
  f->bm25_confidence = zeta * (f->bm25_std_dev / (sqrt(pos)));
  f->bm25_hmean = (double)pos / hmsum;

  free(bmtmp);
  return;
}

void compute_lm_stats(feature_t *f, uint32_t *doclen, posting_t *posting,
                      uint64_t pos, uint64_t clen, uint64_t cf, double *max) {
  uint64_t i;
  uint32_t size = pos;
  uint32_t mid = size / 2;
  uint32_t lq = size / 4;
  uint32_t uq = 3 * size / 4;
  double sum = 0.0, sum_sqrs = 0.0, variance = 0.0;
  double hmsum = 0.0;
  double zeta = 1.960;
  double *bmtmp = (double *)safe_malloc(sizeof(double) * pos);

  for (i = 0; i < pos; i++) {
    double score = 0.0;
    score =
        calculate_lm(posting[i].freq, cf, doclen[posting[i].docid - 1], clen);
    bmtmp[i] = score;
    if (score > *max)
      *max = score;
  }

  qsort(bmtmp, pos, sizeof(double), compare);

  f->lm_median = size % 2 == 0 ? (bmtmp[mid] + bmtmp[mid - 1]) / 2 : bmtmp[mid];
  f->lm_first = size % 2 == 0 ? (bmtmp[lq] + bmtmp[lq - 1]) / 2 : bmtmp[lq];
  f->lm_third = size % 2 == 0 ? (bmtmp[uq] + bmtmp[uq - 1]) / 2 : bmtmp[uq];
  f->lm_max = bmtmp[0];
  f->lm_min = bmtmp[pos - 1];
  for (i = 0; i < pos; i++) {
    sum += bmtmp[i];
    sum_sqrs += bmtmp[i] * bmtmp[i];
    hmsum += 1 / bmtmp[i];
  }

  f->lm_avg = sum / (double)pos;
  f->lm_variance = (sum_sqrs / (double)pos) - f->lm_avg * f->lm_avg;
  f->lm_std_dev = sqrt(variance);
  f->lm_confidence = zeta * (f->lm_std_dev / (sqrt(pos)));
  f->lm_hmean = (double)pos / hmsum;

  free(bmtmp);
  return;
}

void print_feature(feature_t *f) {
  fprintf(stdout, "%s ", f->term);
  fprintf(stdout, "%" SCNu64 " ", f->cf);
  fprintf(stdout, "%" SCNu64 " ", f->cdf);
  fprintf(stdout, "%lf ", f->geo_mean);
  fprintf(stdout, "%lf ", f->bm25_median);
  fprintf(stdout, "%lf ", f->bm25_first);
  fprintf(stdout, "%lf ", f->bm25_third);
  fprintf(stdout, "%lf ", f->bm25_max);
  fprintf(stdout, "%lf ", f->bm25_min);
  fprintf(stdout, "%lf ", f->bm25_avg);
  fprintf(stdout, "%lf ", f->bm25_variance);
  fprintf(stdout, "%lf ", f->bm25_std_dev);
  fprintf(stdout, "%lf ", f->bm25_confidence);
  fprintf(stdout, "%lf ", f->bm25_hmean);
  fprintf(stdout, "%lf ", f->tfidf_median);
  fprintf(stdout, "%lf ", f->tfidf_first);
  fprintf(stdout, "%lf ", f->tfidf_third);
  fprintf(stdout, "%lf ", f->tfidf_max);
  fprintf(stdout, "%lf ", f->tfidf_min);
  fprintf(stdout, "%lf ", f->tfidf_avg);
  fprintf(stdout, "%lf ", f->tfidf_variance);
  fprintf(stdout, "%lf ", f->tfidf_std_dev);
  fprintf(stdout, "%lf ", f->tfidf_confidence);
  fprintf(stdout, "%lf ", f->tfidf_hmean);
  fprintf(stdout, "%lf ", f->lm_median);
  fprintf(stdout, "%lf ", f->lm_first);
  fprintf(stdout, "%lf ", f->lm_third);
  fprintf(stdout, "%lf ", f->lm_max);
  fprintf(stdout, "%lf ", f->lm_min);
  fprintf(stdout, "%lf ", f->lm_avg);
  fprintf(stdout, "%lf ", f->lm_variance);
  fprintf(stdout, "%lf ", f->lm_std_dev);
  fprintf(stdout, "%lf ", f->lm_confidence);
  fprintf(stdout, "%lf ", f->lm_hmean);
  fprintf(stdout, "%lf ", f->pr_median);
  fprintf(stdout, "%lf ", f->pr_first);
  fprintf(stdout, "%lf ", f->pr_third);
  fprintf(stdout, "%lf ", f->pr_max);
  fprintf(stdout, "%lf ", f->pr_min);
  fprintf(stdout, "%lf ", f->pr_avg);
  fprintf(stdout, "%lf ", f->pr_variance);
  fprintf(stdout, "%lf ", f->pr_std_dev);
  fprintf(stdout, "%lf ", f->pr_confidence);
  fprintf(stdout, "%lf ", f->pr_hmean);
  fprintf(stdout, "%lf ", f->be_median);
  fprintf(stdout, "%lf ", f->be_first);
  fprintf(stdout, "%lf ", f->be_third);
  fprintf(stdout, "%lf ", f->be_max);
  fprintf(stdout, "%lf ", f->be_min);
  fprintf(stdout, "%lf ", f->be_avg);
  fprintf(stdout, "%lf ", f->be_variance);
  fprintf(stdout, "%lf ", f->be_std_dev);
  fprintf(stdout, "%lf ", f->be_confidence);
  fprintf(stdout, "%lf ", f->be_hmean);
  fprintf(stdout, "%lf ", f->dph_median);
  fprintf(stdout, "%lf ", f->dph_first);
  fprintf(stdout, "%lf ", f->dph_third);
  fprintf(stdout, "%lf ", f->dph_max);
  fprintf(stdout, "%lf ", f->dph_min);
  fprintf(stdout, "%lf ", f->dph_avg);
  fprintf(stdout, "%lf ", f->dph_variance);
  fprintf(stdout, "%lf ", f->dph_std_dev);
  fprintf(stdout, "%lf ", f->dph_confidence);
  fprintf(stdout, "%lf ", f->dph_hmean);
  fprintf(stdout, "%lf ", f->dfr_median);
  fprintf(stdout, "%lf ", f->dfr_first);
  fprintf(stdout, "%lf ", f->dfr_third);
  fprintf(stdout, "%lf ", f->dfr_max);
  fprintf(stdout, "%lf ", f->dfr_min);
  fprintf(stdout, "%lf ", f->dfr_avg);
  fprintf(stdout, "%lf ", f->dfr_variance);
  fprintf(stdout, "%lf ", f->dfr_std_dev);
  fprintf(stdout, "%lf ", f->dfr_confidence);
  fprintf(stdout, "%lf ", f->dfr_hmean);
  fprintf(stdout, "\n");
  return;
}

static void print_usage(const char *program) {
  fprintf(stderr, "USAGE: %s [options]\n", program);
  fprintf(stderr, "  -i --inv-file file  Inverted file input\n");
  fprintf(stderr, "  -d --doclen file    Document Lengths input\n");
  fprintf(stderr, "  -c --collen         Length of the collection\n");
  fprintf(stderr, "  -v --verbose        Dump debugging information\n");
  fprintf(stderr, "  -h --help           Display usage information\n");
  fprintf(stderr, "\n");
  fprintf(stderr, "EXAMPLE: %s -i text.inv -o doc_lens.txt -c 40541601698\n",
          program);
  fprintf(stderr, "\n");
  return;
}

/** Initialize resources for the command line options **/
cmd_opt_t *init_cmd_opt_t(void) {
  cmd_opt_t *rv = NULL;
  rv = (cmd_opt_t *)safe_malloc(sizeof(cmd_opt_t));

  rv->idx_file_name = NULL;
  rv->dlen_file_name = NULL;
  rv->collen = 0;
  rv->verbose = FALSE;

  return (rv);
}

/** Free all resources allocated for the command line options **/
void destroy_cmd_opt_t(cmd_opt_t *noret) {
  if (!noret) {
    fprintf(stderr, "WARNING: command line options does not exist!\n");
    return;
  }
  if (noret->idx_file_name) {
    free(noret->idx_file_name);
  }
  if (noret->dlen_file_name) {
    free(noret->dlen_file_name);
  }
  free(noret);
}

cmd_opt_t *parse_args(int argc, char **argv) {
  cmd_opt_t *cmd_line = NULL;
  int opt = 0;
  int lidx = 0;
  int i;

  static const char *optstr = "d:i:c:vh?";

  static const struct option options_table[] = {
      {"inv-file", required_argument, NULL, 'i'},
      {"dlen-file", required_argument, NULL, 'd'},
      {"collen", required_argument, NULL, 'c'},
      {"verbose", no_argument, NULL, 'v'},
      {"help", no_argument, NULL, 'h'},
      {NULL, no_argument, NULL, 0},
  };

  if (argc == 1) {
    print_usage(argv[0]);
    exit(EXIT_FAILURE);
  }

  cmd_line = init_cmd_opt_t();

  opt = getopt_long(argc, argv, optstr, options_table, &lidx);
  while (opt != -1) {
    switch (opt) {
    case 'i':
      cmd_line->idx_file_name = safe_strdup(optarg);
      break;
    case 'd':
      cmd_line->dlen_file_name = safe_strdup(optarg);
      break;
    case 'c':
      cmd_line->collen = strtoull(optarg, NULL, 10);
      break;
    case 'v':
      cmd_line->verbose = TRUE;
      break;

    /* Fall through */
    case 0:
      fprintf(stderr, "WARN: No short argument <%s>\n",
              options_table[lidx].name);
    case 'h':
    case '?':
      print_usage(argv[0]);
      exit(EXIT_FAILURE);
      break;
    }
    opt = getopt_long(argc, argv, optstr, options_table, &lidx);
  }

  /*
  if (cmd_line->verbose == TRUE) {
    dump_cmd_line_opts (cmd_line);
  } */
  /** Catch unparsed command line input **/
  for (i = optind; i < argc; ++i) {
    fprintf(stderr, "WARN: Option Ignored <%s>\n", argv[i]);
  }

  return (cmd_line);
}

int main(int argc, char **argv) {
  FILE *in_file = NULL;
  uint64_t cf = 0;
  char term[4096];
  uint64_t pos = 0;
  uint32_t docid = 0;
  uint32_t df = 0, i = 0;
  int done = 0;
  int freq = 0;
  double tfidf_max = 0.0;
  double bm25_max = 0.0;
  double pr_max = 0.0;
  double be_max = 0.0;
  double dfr_max = 0.0;
  double dph_max = 0.0;
  double lm_max = -DBL_MAX;
  /* ONLY for CLUEWEB09B! */
  // uint64_t clen = 40541601698;
  uint64_t clen = 0;
  cmd_opt_t *cmd_line = NULL;

  int ndocs = 0;
  uint32_t *doclen = NULL;
  double avg_dlen = 0.0;

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
  while (fscanf(in_file, "%s", term) == 1) {
    feature_t *f = safe_malloc(sizeof(feature_t));
    int sz = 0;
    fscanf(in_file, "%" SCNu64, &cf);
    fscanf(in_file, "%" SCNu64, &pos);

    f->cf = cf;
    f->cdf = pos;
    sz = strlen(term);
    if (sz < 1023) {
      // printf ("-");
      strcpy(f->term, term);
      // printf ("--");
    } else {
      fprintf(stderr, "ERROR term <%s> too long <%d>.", term, sz);
      exit(EXIT_FAILURE);
    }

    /* Min count is set to 4 or IQR computation goes boom. */
    if (pos >= 4) {
      posting_t *posting = safe_malloc(sizeof(posting_t) * pos);
      for (i = 0; i < pos; i++) {
        fscanf(in_file, "%" SCNu32 ":", &docid);
        fscanf(in_file, "%" SCNu32, &df);
        posting[i].docid = docid;
        posting[i].freq = df;
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
