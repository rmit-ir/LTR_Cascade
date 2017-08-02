"""
Trains gradient boosting regression trees, boosting decision trees and lambdamart using xgboost
allows for parameter exploration using cross validation

roi blanco
"""
from __future__ import print_function

import ast
import baker
import logging
import math
import numpy as np
import scipy.sparse
from sklearn.externals import joblib
from tqdm import trange

import core
from core.cascade import load_data_file, load_data, load_model, save_model
from core.utils import group_counts
from core.metrics import test_all, test_ndcg


# This is for suppressing the warning messages from xgboost
import warnings
warnings.filterwarnings("ignore")

import xgboost as xgb  # noqa


def load_DMatrix(x, y, cache_file=None):
    import os.path

    if cache_file is not None:
        if os.path.exists(cache_file):
            logging.info("Load cache '{}'".format(cache_file))
            return xgb.DMatrix(cache_file)
        else:
            logging.info("Write to cache '{}'".format(cache_file))
            dm = xgb.DMatrix(x, y)
            dm.save_binary(cache_file)
            return dm
    else:
        logging.info("No cache")
        return xgb.DMatrix(x, y)


def train(train_data, valid_data, score_function, class_weights,
          params, trees, nodes, features=None, set_classes=False,
          train_cache=None, valid_cache=None):
    x_train, y_train, qid_train, _ = train_data
    x_valid, y_valid, qid_valid, _ = valid_data

    if features is None:
        # dtrain = xgb.DMatrix(x_train, y_train)
        dtrain = load_DMatrix(x_train, y_train, train_cache)

        dvalid = None
        if x_valid is not None:
            dvalid = xgb.DMatrix(x_valid, y_valid)
            dvalid = load_DMatrix(x_valid, y_valid, valid_cache)
    else:
        non_open_features = np.setdiff1d(np.arange(x_train.shape[1]), features)

        # hide non-open features
        x_train_prime = x_train.copy()
        x_train_prime[:, non_open_features] = 0
        # dtrain = xgb.DMatrix(x_train_prime, y_train)
        dtrain = load_DMatrix(x_train_prime, y_train, train_cache)

        dvalid = None
        if x_valid is not None:
            x_valid_prime = x_valid.copy()
            x_valid_prime[:, non_open_features] = 0
            # dvalid = xgb.DMatrix(x_valid_prime, y_valid)
            dvalid = load_DMatrix(x_valid_prime, y_valid, valid_cache)

    if dvalid:
        watchlist = [(dvalid, 'eval'), (dtrain, 'train')]  # remove if the output is too verbose
    else:
        watchlist = [(dtrain, 'train')]

    if set_classes:
        params['num_class'] = np.unique(y_train).shape[0]

    best_sc = -1000
    best_params = None
    bst = None

    def tqdm_integration(tbar):
        def callback(env):
            tbar.update()
            tbar.set_description(' '.join(['{}:{}'.format(k, v) for k, v in env.evaluation_result_list]))
            if env.iteration == env.end_iteration:
                tbar.close()
        return callback

    for t in trees:
        for n in nodes:
            params['max_depth'] = int(math.ceil(math.log(n, 2)))  # NOTE: max_depth is automatically set
            logging.info('Training with %i trees and %i depth' % (t, n))
            logging.info('Params %s' % params)

            # model = xgb.train(params, dtrain, t, watchlist)
            with trange(t) as tbar:
                model = xgb.train(params, dtrain, t, watchlist, verbose_eval=False, callbacks=[tqdm_integration(tbar)])

            if dvalid:
                predictions = score_function(model.predict(dvalid), class_weights)
                sc = test_ndcg(predictions, y_valid, qid_valid)  # this groups the validation queries each time, redo
            else:
                predictions = score_function(model.predict(dtrain), class_weights)
                sc = test_ndcg(predictions, y_train, qid_train)

            if sc > best_sc:
                bst = model
                best_sc = sc
                best_params = params.copy()
                best_params['n_trees'] = t
                best_params['n_nodes'] = n  # NOTE: for reference

    if hasattr(bst, 'set_attr'):
        bst.set_attr(**{k: str(v) for k, v in best_params.items()})
    return bst


def add_original_order_as_feature(data):
    x, _, qid, _ = data
    feature = np.concatenate([np.linspace(0, 1, c + 1)[-1:0:-1] for c in group_counts(qid)])
    sparse_feature = scipy.sparse.csr_matrix(feature.reshape((feature.size, 1)))
    return scipy.sparse.hstack((x, sparse_feature))


def predict(model, test_data, score_function, class_weights, output_trec_run=None):
    x_test, y_test, qid_test, docno_test = test_data

    dtest = xgb.DMatrix(x_test)
    preds = score_function(model.predict(dtest), class_weights)
    test_metrics = test_all(preds, y_test, qid_test, 1)
    print(test_metrics)

    if output_trec_run:
        with open(output_trec_run, 'w') as output:
            core.cascade.print_trec_run(output, preds, y_test, qid_test, docno_test)
        print('Result saved to %s' % output_trec_run)


def train_tree_ranker(train_file, validation_file, test_file, model_prefix,
                      score_function, params, trees, nodes, set_classes=False,
                      add_original_order=False):
    train_data, valid_data, test_data = load_data(train_file, validation_file, test_file)

    if add_original_order:
        # FIXME: quick hack
        logging.info('The original-order hack is applied to all data')
        train_data = (add_original_order_as_feature(train_data), train_data[1], train_data[2])
        if valid_data[0] is not None:
            valid_data = (add_original_order_as_feature(valid_data), valid_data[1], valid_data[2])
        if test_data[0] is not None:
            test_data = (add_original_order_as_feature(test_data), test_data[1], test_data[2])

    class_weights = core.get_class_weights(train_data[1])
    model = train(train_data, valid_data, score_function, class_weights,
                  params, trees, nodes, set_classes=set_classes)
    if model_prefix:
        save_model(model, model_prefix)
    predict(model, test_data, score_function, class_weights)


@baker.command(name='train_GBRT')
def do_train_GBRT(train_file, validation_file, test_file, model_prefix=None, learning_rate="0.1",
                  silent=True, subsample="0.5", trees="[5,10,20,50,1000]", nodes="[32]",
                  add_original_order=False):
    """Train a gradient-boosting regression tree"""
    params = {'eta': ast.literal_eval(learning_rate),
              'silent': silent,
              'objective': 'reg:linear',
              'eval_metric': 'rmse',
              'subsample': ast.literal_eval(subsample)}
    train_tree_ranker(train_file, validation_file, test_file, model_prefix,
                      core.get_score, params, ast.literal_eval(trees),
                      ast.literal_eval(nodes), set_classes=False,
                      add_original_order=add_original_order)


@baker.command(name='train_GBDT')
def do_train_GBDT(train_file, validation_file, test_file, model_prefix=None, learning_rate="0.1",
                  silent=True, subsample="0.5", trees="[5,10,20,50,1000]", nodes="[32]",
                  add_original_order=False):
    """Train a gradient-boosting decision tree"""
    params = {'eta': ast.literal_eval(learning_rate),
              'silent': silent,
              'objective': 'multi:softprob',
              'eval_metric': 'mlogloss',
              'subsample': ast.literal_eval(subsample)}
    train_tree_ranker(train_file, validation_file, test_file, model_prefix,
                      core.get_score_multiclass, params, ast.literal_eval(trees),
                      ast.literal_eval(nodes), set_classes=True,
                      add_original_order=add_original_order)


@baker.command(name='train_LambdaMART')
def do_train_LambdaMART(train_file, validation_file, test_file, model_prefix=None, learning_rate="0.1",
                        silent=True, subsample="0.5", trees="[5,10,20,50,1000]", nodes="[32]",
                        add_original_order=False):
    """Train a LambdaMART model"""
    params = {'eta': ast.literal_eval(learning_rate),
              'silent': silent,
              'objective': 'rank:pairwise',
              'eval_metric': 'rmse',
              'subsample': ast.literal_eval(subsample)}
    train_tree_ranker(train_file, validation_file, test_file, model_prefix,
                      core.get_score, params, ast.literal_eval(trees),
                      ast.literal_eval(nodes), set_classes=False,
                      add_original_order=add_original_order)


@baker.command(name='predict_GBRT')
def do_predict_GBRT(test_file, model_file, output_trec_run=None, add_original_order=False):
    """Run prediction with a saved model"""
    test_data = load_data_file(test_file)
    if add_original_order:
        test_data = (add_original_order_as_feature(test_data), test_data[1], test_data[2])
    model = load_model(model_file)
    predict(model, test_data, core.get_score, None, output_trec_run=output_trec_run)


@baker.command(name='predict_GBDT')
def do_predict_GBDT(test_file, model_file, output_trec_run=None, add_original_order=False):
    """Run prediction with a saved model"""
    test_data = load_data_file(test_file)
    if add_original_order:
        test_data = (add_original_order_as_feature(test_data), test_data[1], test_data[2])
    model = load_model(model_file)
    class_weights = core.get_class_weights(test_data[1])  # FIXME: shouldn't peek into this
    predict(model, test_data, core.get_score_multiclass, class_weights,
            output_trec_run=output_trec_run)


@baker.command(name='predict_LambdaMART')
def do_predict_LambdaMART(test_file, model_file, output_trec_run=None, add_original_order=False):
    """Run prediction with a saved model"""
    test_data = load_data_file(test_file)
    if add_original_order:
        test_data = (add_original_order_as_feature(test_data), test_data[1], test_data[2])
    model = load_model(model_file)
    predict(model, test_data, core.get_score, None, output_trec_run=output_trec_run)


@baker.command
def dump_importance(model_file, max_fid, importance_type='weight'):
    """Dump feature importance scores.

    Args:
        model_file: the model file
        max_fid: max. feature id
        importance_type: 'weight' or 'gain'
    """
    bst = joblib.load(model_file)
    score_map = bst.get_score(importance_type=importance_type)
    score_map = {int(k[1:]): float(v) for k, v in score_map.items()}
    for i in range(int(max_fid)):
        print(int(score_map.get(i, 0)))


@baker.command
def info(model_file, costs_file=None):
    bst = joblib.load(model_file)
    fids = sorted([int(k[1:]) for k in bst.get_fscore()])

    print('params', vars(bst))
    if hasattr(bst, 'attributes'):
        print('attributes', bst.attributes())
    print('n_features', len(fids))
    print('feature list', fids)

    if costs_file:
        from core.cascade import load_costs_data
        costs, _ = load_costs_data(costs_file, None, max(fids) + 1)
        mask = np.zeros(costs.size, dtype=int)
        np.put(mask, fids, 1)
        print('cost %d' % np.dot(costs, mask))


if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                        level=logging.INFO)
    baker.run()
