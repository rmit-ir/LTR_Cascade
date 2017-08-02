from __future__ import print_function

import logging
import math
import numpy as np

from sklearn.datasets import load_svmlight_file
from sklearn.externals import joblib
from . import io

import warnings
warnings.filterwarnings("ignore")


def load_data_file(filename):
    if filename.endswith('.npz'):
        return io.load_npz(filename)
    else:
        return load_svmlight_file(filename, query_id=True)


def load_data(train_file, validation_file, test_file, scaler=None):
    """Prepare training/validation/test data."""
    x_train, y_train, qid_train, docno_train = load_data_file(train_file)
    logging.info('Load %s: x_train %s, y_train %s, qid_train %s' %
                 (train_file, x_train.shape, y_train.shape, qid_train.shape))

    x_test, y_test, qid_test, docno_test = load_data_file(test_file)
    logging.info('Load %s: x_test %s, y_test %s, qid_test %s' %
                 (test_file, x_test.shape, y_test.shape, qid_test.shape))
    assert x_test.shape[1] == x_train.shape[1]

    x_valid, y_valid, qid_valid, docno_valid = None, None, None, None
    if validation_file:
        x_valid, y_valid, qid_valid, docno_valid = load_data_file(validation_file)
        logging.info('Load %s: x_valid %s, y_valid %s, qid_valid %s' %
                     (validation_file, x_valid.shape, y_valid.shape, qid_valid.shape))
        assert x_valid.shape[1] == x_train.shape[1]

    if scaler:
        scaler.fit_transform(x_train)
        scaler.transform(x_test)
        if x_valid is not None:
            scaler.transform(x_valid)

    y_train.flags.writeable = False
    qid_train.flags.writeable = False

    y_test.flags.writeable = False
    qid_test.flags.writeable = False

    if x_valid is not None:
        y_valid.flags.writeable = False
        qid_valid.flags.writeable = False

    return ((x_train, y_train, qid_train, docno_train),
            (x_valid, y_valid, qid_valid, docno_valid),
            (x_test, y_test, qid_test, docno_test))


def load_costs_data(costs_file, importance_file, n_features):
    """Load costs/importance data."""
    costs = np.loadtxt(costs_file) if costs_file else np.ones(n_features)
    logging.info('Load %s: costs %s' % (costs_file, costs.shape))

    importance = np.loadtxt(importance_file) if importance_file else np.ones(n_features)
    logging.info('Load %s: importance %s' % (importance_file, importance.shape))

    if costs.shape[0] > n_features:
        costs = np.resize(costs, n_features)
        logging.info('costs resized to match n_features %i' % n_features)

    if importance.shape[0] > n_features:
        importance = np.resize(importance, n_features)
        logging.info('importance resized to match n_features %i' % n_features)

    costs.flags.writeable = False
    importance.flags.writeable = False

    return costs, importance


def load_model(filename):
    """Load model from file."""
    return joblib.load(filename)
    print('Model loaded from %s' % filename)


def save_model(model, filename):
    """Save the model to file."""
    joblib.dump(model, filename)
    print('Model saved to %s' % filename)


def predict(cascade, x, qid, score_update):
    """Run prediciton using the cascade"""
    state = init_predict(x)
    results = []
    for stage in cascade:
        new_state = partial_predict(stage, state, x, qid, score_update)
        results.append(new_state)
        state = new_state
    return results


def init_predict(x):
    return {'preds': np.zeros(x.shape[0], dtype=float),
            'indexes': np.arange(x.shape[0], dtype=int),
            'extract_counts': np.zeros(x.shape[1], dtype=int)}


def partial_predict(stage, state, x, qid, score_update):
    """Run partial prediction by executing one cascade stage"""
    prune, model = stage

    if prune is None:
        indexes = state['indexes'].copy()
    else:
        pruned = []
        for a, b in group_offsets(qid[state['indexes']]):
            idx = state['indexes'][a:b]
            ranked_idx = idx[np.argsort(state['preds'][idx])[::-1]]
            pruned.extend(prune(ranked_idx))
        indexes = np.array(sorted(pruned))

    # extracted features will not receive more counts
    new_counts = (state['extract_counts'] == 0).astype(int) * model.get_feature_mask() * indexes.size
    extract_counts = state['extract_counts'] + new_counts

    scores = model.predict(x[indexes])
    preds = score_update(state['preds'], indexes, scores)
    return {'preds': preds, 'indexes': indexes, 'extract_counts': extract_counts}


def print_trec_run(output, preds, y, qid, docno=None, run_id='exp'):
    for a, b in group_offsets(qid):
        sim = preds[a:b].copy()
        if docno is None:
            docno_string = ['%s.%d' % (qid[a], i) for i in range(1, b - a + 1)]
        else:
            docno_string = docno[a:b]
        ranked_list = sorted(zip(docno_string, sim), key=lambda x: x[1], reverse=True)
        for rank, (d, s) in enumerate(ranked_list, 1):
            output.write('%s Q0 %s %i %f %s\n' % (qid[a], d, rank, s, run_id))


def group_counts(arr):
    d = np.ones(arr.size, dtype=int)
    d[1:] = (arr[:-1] != arr[1:]).astype(int)
    return np.diff(np.where(np.append(d, 1))[0])


def group_offsets(arr):
    """Return a sequence of start/end offsets for the value subgroups in the input"""
    d = np.ones(arr.size, dtype=int)
    d[1:] = (arr[:-1] != arr[1:]).astype(int)
    idx = np.where(np.append(d, 1))[0]
    return zip(idx, idx[1:])


# Score update classes
#
class AdditiveUpdate(object):
    def __call__(self, preds, indexes, update):
        new_preds = preds.copy()
        new_preds[indexes] = new_preds[indexes] + update  # to work around the qfunc 'add' bug
        return new_preds


class UpshiftUpdate(object):
    def __init__(self, gap):
        self.gap = gap

    def __call__(self, preds, indexes, update):
        diff = max(0, preds.max() + self.gap - update.min())  # mind the gap
        new_preds = preds.copy()
        new_preds[indexes] = update + diff
        return new_preds


class ResetUpdate(UpshiftUpdate):
    def __call__(self, preds, indexes, update):
        return super(ResetUpdate, self).__call__(np.zeros_like(preds), indexes, update)


# Prune
#
class Prune(object):
    def __init__(self, rank=None, beta=None):
        self.rank = rank
        self.beta = beta

    def __call__(self, arr):
        if self.rank:
            cutoff = self.rank
        elif self.beta:
            cutoff = int(math.ceil(len(arr) * self.beta))
        else:
            cutoff = None
        return arr[:cutoff]


# Model classes
#
class SGDClassifierModel(object):
    def __init__(self, model):
        self.model = model

    def get_feature_mask(self):
        return (self.model.coef_[0] != 0).astype(int)

    def predict(self, x):
        return self.model.decision_function(x)


class LinearModel(object):
    def __init__(self, coef):
        self.coef = coef.copy()

    def get_feature_mask(self):
        return (self.coef != 0).astype(int)

    def predict(self, x):
        return np.dot(x, self.coef)


class TreeModel(object):
    def __init__(self, model, score_function, class_weights, n_features):
        self.model = model
        self.score_function = score_function
        self.class_weights = class_weights
        self.n_features = n_features

    def get_feature_mask(self):
        mask = np.zeros(self.n_features, dtype=int)
        for k in self.model.get_score():
            mask[int(k[1:])] = 1
        return mask

    def predict(self, x):
        import xgboost as xgb

        dm = xgb.DMatrix(x.toarray())
        return self.score_function(self.model.predict(dm), self.class_weights)


class SVMModel(object):
    def __init__(self, model, score_function, class_weights, n_features):
        self.model = model
        self.score_function = score_function
        self.class_weights = class_weights
        self.n_features = n_features

    def get_feature_mask(self):
        return (self.model.coef_[0] != 0).astype(int)

    def predict(self, x):
        return self.score_function(self.model.predict(x), self.class_weights)
