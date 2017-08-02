from __future__ import print_function

import ast
import baker
import logging
import math
import numpy as np

from sklearn.preprocessing import MaxAbsScaler
from tqdm import tqdm

import core
from core.cascade import load_data, load_data_file, load_costs_data, load_model, save_model, group_counts, group_offsets
from core.metrics import test_all, test_ndcg


def _predict(cascade, x, qid, return_stages=False):
    """Run prediciton"""
    preds, indexes = _init_predict(x)

    if return_stages:
        stagewise_results = []
        for stage in cascade:
            result = _partial_predict(stage, preds, indexes, x, qid)
            stagewise_results.append(result)
            preds, indexes = result
        return stagewise_results
    else:
        for stage in cascade:
            preds, indexes = _partial_predict(stage, preds, indexes, x, qid)
        return preds, indexes


def _init_predict(x):
    """Initialze the predictions and indexes"""
    preds = np.full(x.shape[0], -1, dtype=float)
    indexes = np.arange(x.shape[0], dtype=int)
    return preds, indexes


def _partial_predict(stage, preds, indexes, x, qid):
    """Run partial prediction by executing one cascade stage"""
    prune, model = stage
    if prune:
        new_indexes = []
        for a, b in group_offsets(qid[indexes]):
            idx = indexes[a:b]
            ranked_idx = idx[np.argsort(preds[idx])[::-1]]
            cutoff = int(math.ceil(prune['beta'] * (b - a)))  # prevent generating empty ranked lists
            if cutoff == 0:
                print(ranked_idx, prune['beta'], b - a)
            new_indexes.extend(ranked_idx[:cutoff])
        new_indexes = np.array(sorted(new_indexes))
    else:
        new_indexes = indexes.copy()

    new_preds = preds.copy()
    new_scores = np.dot(x[new_indexes], model)
    new_preds[new_indexes] = new_preds[new_indexes] + new_scores  # to work around the numpy qfunc 'add' bug
    return new_preds, new_indexes


def predict(cascade, test_data, costs, output_trec_run=None, output_eval=None):
    """Run prediction using the cascade."""
    x, y, qid, docno = test_data
    x = x.toarray()

    # NOTE: the cost-aware evaluation protocol is implemented differently here.
    # `extracted_count` is currently stagewise and does not keep track of
    # previously extracted features.  So to compute the total cascade cost, we
    # need to add all the stagewise costs together.
    cost_spent_weighted = 0

    stagewise_results = _predict(cascade, x, qid, return_stages=True)
    for i, ((prune, model), (preds, indexes)) in enumerate(zip(cascade, stagewise_results)):
        test_metrics = test_all(preds, y, qid, 1)
        print('stage %i: '
              'test ERR@5/10/20 %0.4f/%0.4f/%0.4f, '
              'test NDCG@5/10/20 %0.4f/%0.4f/%0.4f, '
              'test P@5/10/20 %0.4f/%0.4f/%0.4f' %
              (i,
               test_metrics['err@5'], test_metrics['err@10'], test_metrics['err@20'],
               test_metrics['ndcg@5'], test_metrics['ndcg@10'], test_metrics['ndcg@20'],
               test_metrics['p@5'], test_metrics['p@10'], test_metrics['p@20']))

        n_used_features = len(np.flatnonzero(model))
        n_active_docs = len(indexes)
        extracted_count = (model != 0).astype(float) * len(indexes)

        # NOTE: note the +=
        cost_spent_weighted += np.sum(costs * extracted_count)

        print('          weighted L1 %f, cascade features %i, num docs %i, cascade cost %0.2f' %
              (np.nan,
               n_used_features,
               n_active_docs,
               cost_spent_weighted / float(x.shape[0])))

    if output_trec_run:
        with file(output_trec_run, 'wb') as output:
            core.cascade.print_trec_run(output, stagewise_results[-1][0], y, qid, docno)
        logging.info('TREC run saved to %s' % output_trec_run)


def train(train_data, valid_data, costs, importance, n_stages=0,
          gamma=0.1, beta_values=[1.0], use_query_features=False):
    """Learn one ranker with SGD and L1 regularization.

    Args:
        n_stages: number of rankers in the cascade
        strategies: a dict of callback functions
    """
    x_train, y_train, qid_train, _ = train_data
    x_train = x_train.toarray()

    # FIXME: validation data manually turned off
    #        for weird reasons, validation based early stopping doesn't work well
    valid_data = None

    if valid_data:
        x_valid, y_valid, qid_valid, _ = valid_data
        x_valid = x_valid.toarray()

    n_queries = np.unique(qid_train).shape[0]
    n_features = x_train.shape[1]
    n_stages = n_stages or n_features  # n_stages = n_features if set to None

    weights = np.ones(n_queries, dtype=float) / n_queries
    C_cascade = np.zeros(n_queries, dtype=float)
    cascade = []

    # NOTE: gamma is normalized by the maximum cost times the number of docs
    max_cost = max(np.max(costs), 1)
    C_normalizer = float(max_cost) * x_train.shape[0]

    best_perf_train, best_perf_valid = -np.inf, -np.inf
    best_cascade = None

    # The cascade doesn't like query features...
    features = []
    if use_query_features:
        for j, _ in enumerate(costs):
            features.append(j)
    else:
        for j, _ in enumerate(costs):
            for a, b in group_offsets(qid_train):
                if (x_train[a:b, j] != x_train[a, j]).any():
                    features.append(j)
                    break

    used_fids = []
    preds, indexes = _init_predict(x_train)

    for _ in range(n_stages):
        best_weighted_perf = -np.inf
        best_stage = None

        for k in tqdm(features, 'scan through features'):
            if k in used_fids:
                continue

            weak_ranker = np.zeros(n_features, dtype=float)
            weak_ranker[k] = 1
            # for beta in np.linspace(0, 1, 4)[1:]:
            for beta in beta_values:
                prune = {'beta': beta}
                new_preds, new_indexes = _partial_predict((prune, weak_ranker),
                                                          preds, indexes, x_train, qid_train)
                # Eq. (6) in Wang et al. (2011)
                E = np.array(test_ndcg(new_preds, y_train, qid_train, average=False))
                C = costs[k] * group_counts(qid_train[new_indexes]) / C_normalizer

                try:
                    term1 = np.sum(weights * E / (1 - gamma * C))  # phi_t
                    term2 = np.sum(weights / (1 - gamma * C))
                except Exception as e:
                    print(weights.shape, E.shape, C.shape)
                    print(np.unique(qid_train[new_indexes]).shape)
                    raise e

                weighted_perf = term1 ** 2 - term2 ** 2
                if weighted_perf > best_weighted_perf:
                    best_stage = {'J': prune, 'H': weak_ranker, 'E': E, 'C': C, 'fid': k}
                    best_weighted_perf = weighted_perf

        if not best_stage:
            break

        S = best_stage

        alpha = 0.5 * math.log(
            np.sum(weights * (1 + S['E']) / (1 - gamma * S['C'])) /
            np.sum(weights * (1 - S['E']) / (1 - gamma * S['C']))
        )
        S['alpha'] = alpha
        S['H'] *= alpha

        print('J:', S['J'], 'fid:', S['fid'] + 1)  # the internal fid is 0 based
        print('H:', S['H'].nonzero(), '(values)', S['H'][S['H'].nonzero()])

        stage = (S['J'], S['H'])
        cascade.append(stage)

        # update feature sets and cascade cost
        used_fids.append(S['fid'])
        C_cascade = C_cascade + S['C']

        # update predictions and indexes
        # preds, indexes = _predict(cascade, x_train, qid_train)
        new_preds, new_indexes = _partial_predict(stage, preds, indexes, x_train, qid_train)
        print('preds', preds[:5], 'new_preds', new_preds[:5])

        preds = new_preds
        indexes = new_indexes

        # update cascade effectiveness
        E_cascade = np.array(test_ndcg(preds, y_train, qid_train, average=False))

        perf_train = E_cascade.mean()

        if valid_data:
            perf_valid = test_ndcg(_predict(cascade, x_valid, qid_valid)[0],
                                   y_valid, qid_valid, average=True)
        else:
            perf_valid = np.nan

        print('train ndcg %0.4f, valid ndcg %0.4f' % (perf_train, perf_valid))
        if perf_train <= best_perf_train:  # NOTE: stop early when performance plateaued
            break

        best_perf_train = perf_train

        if valid_data:
            if perf_valid > best_perf_valid:
                best_perf_valid = perf_valid
                best_cascade = list(cascade)
        else:
            best_cascade = list(cascade)

        new_weights = np.exp(-E_cascade + gamma * C_cascade)
        weights = new_weights / new_weights.sum()
        # print('weight', weights[:10])

    return best_cascade


def build_wlm11_cascade(train_file, validation_file, test_file, costs_file=None,
                        importance_file=None, model_prefix=None, **kwargs):
    """Train a cascade over a partition of disjoint feature sets."""
    train_data, valid_data, test_data = load_data(
        train_file, validation_file, test_file, scaler=MaxAbsScaler(copy=False))
    costs, importance = load_costs_data(
        costs_file, importance_file, n_features=train_data[0].shape[1])

    # NOTE: costs has to be untainted (make copy before passing it to functions)
    cascade = train(train_data, valid_data, costs.copy(), importance.copy(), **kwargs)

    if model_prefix:
        save_model(cascade, model_prefix)
    predict(cascade, test_data, costs.copy())


@baker.command(name='train')
def WLM11(train_file, validation_file, test_file, costs_file=None, importance_file=None,
          model_prefix=None, n_stages=0, gamma="0.1", beta_values="[0.5, 0.33, 0.25, 0.2, 0.1]",
          use_query_features=False):
    """Train a cascade accoring to the algorithm in Wang et al. (2011)"""

    build_wlm11_cascade(train_file, validation_file, test_file, costs_file, importance_file,
                        model_prefix=model_prefix, n_stages=n_stages,
                        gamma=ast.literal_eval(gamma), beta_values=ast.literal_eval(beta_values),
                        use_query_features=use_query_features)


@baker.command(name='predict')
def do_predict(test_file, costs_file, model_file, output_trec_run=None, output_eval=None, train_file=None):
    """Run prediction with a saved cascade"""
    test_data = load_data_file(test_file)
    costs, _ = load_costs_data(costs_file, None, n_features=test_data[0].shape[1])

    cascade = load_model(model_file)

    # FIXME: scaler needs to be saved along the cascade
    if train_file:
        train_data = load_data_file(train_file)
        scaler = MaxAbsScaler(copy=False)
        scaler.fit(train_data[0])
        scaler.transform(test_data[0])
        logging.info('Data scaled')

    if 'scaler' in cascade:
        cascade['scaler'].transform(test_data[0])
    predict(cascade, test_data, costs,
            output_trec_run=output_trec_run, output_eval=output_eval)


@baker.command(name='info')
def do_info(model_file):
    cascade = load_model(model_file)
    for i, (prune, stage) in enumerate(cascade, 1):
        k = np.flatnonzero(stage)
        print('stage', i, 'prune', prune, 'fid', k + 1, 'weight', stage[k])  # fid is 0 based


if __name__ == "__main__":
    logging.basicConfig(
        format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    baker.run()
