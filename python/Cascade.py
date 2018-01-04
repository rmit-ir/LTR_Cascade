from __future__ import print_function

import ast
import baker
import logging
import numpy as np

from scipy.sparse import csr_matrix
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import MaxAbsScaler

import core
from core.cascade import load_data_file, load_data, load_costs_data, load_model, save_model
from core.cascade import Prune, TreeModel, SVMModel, SGDClassifierModel, group_offsets
from core.metrics import test_all


# TODO: batch per and not per example (group by query-id using utils.py and sample per group)
def batch_generator(X, y, batch_size, samples_per_epoch):
    """Generate mini-batches."""
    number_of_batches = samples_per_epoch / batch_size
    shuffle_index = np.arange(np.shape(y)[0])
    np.random.shuffle(shuffle_index)
    X = X[shuffle_index, :]
    y = y[shuffle_index]

    for i in range(number_of_batches):
        index_batch = shuffle_index[batch_size * i:batch_size * (i + 1)]
        X_batch = X[index_batch, :]
        if isinstance(X_batch, csr_matrix):
            X_batch = X_batch.todense()
        y_batch = y[index_batch]
        yield np.array(X_batch), y_batch


def predict(cascade, test_data, costs, output_trec_run=None, output_eval=None):
    """Run prediction using the cascade."""
    x, y, qid, docno = test_data
    # if 'scaler' in cascade:
    #     cascade['scaler'].transform(x)

    states = core.cascade.predict(cascade['stages'], x, qid, cascade['score_update'])
    eval_results = {}

    for i, state in enumerate(states, 1):
        test_metrics = test_all(state['preds'], y, qid, 1)  # NOTE: rel was set to 1
        print('stage %i: '
              'test ERR@5/10/20 %0.4f/%0.4f/%0.4f, '
              'test NDCG@5/10/20 %0.4f/%0.4f/%0.4f, '
              'test P@5/10/20 %0.4f/%0.4f/%0.4f' %
              (i,
               test_metrics['err@5'], test_metrics['err@10'], test_metrics['err@20'],
               test_metrics['ndcg@5'], test_metrics['ndcg@10'], test_metrics['ndcg@20'],
               test_metrics['p@5'], test_metrics['p@10'], test_metrics['p@20']))

        n_used_features = len(np.flatnonzero(state['extract_counts']))
        n_active_docs = len(state['indexes'])
        cost_spent_weighted = np.sum(costs * state['extract_counts'])
        print('          weighted L1 %f, cascade features %i, num docs %i, cascade cost %0.2f' %
              (np.nan,
               n_used_features,
               n_active_docs,
               cost_spent_weighted / float(x.shape[0])))

        name = 'stage%i' % i if i < len(states) else 'cascade'
        for m in ['err', 'ndcg', 'map', 'ndcg@5', 'ndcg@10', 'ndcg@20',
                  'p@5', 'p@10', 'p@20', 'err@5', 'err@10', 'err@20']:
            eval_results['%s_%s' % (name, m)] = test_metrics[m]
        eval_results['%s_n_features' % name] = n_used_features
        eval_results['%s_n_docs' % name] = n_active_docs
        eval_results['%s_cost' % name] = cost_spent_weighted
        eval_results['%s_cost_per_doc' % name] = cost_spent_weighted / float(x.shape[0])

    if output_eval:
        with open(output_eval, 'w') as output:
            for i, _ in enumerate(states, 1):
                name = 'stage%i' % i if i < len(states) else 'cascade'
                for m in ['n_features', 'n_docs']:
                    measure = '%s_%s' % (name, m)
                    output.write('%-24s%i\n' % (measure, eval_results[measure]))
                for m in ['err', 'ndcg', 'map', 'err@5', 'err@10', 'err@20',
                          'ndcg@5', 'ndcg@10', 'ndcg@20', 'p@5', 'p@10', 'p@20',
                          'cost', 'cost_per_doc']:
                    measure = '%s_%s' % (name, m)
                    output.write('%-24s%0.4f\n' % (measure, eval_results[measure]))
        logging.info('Eval result saved to %s' % output_eval)

    if output_trec_run:
        with open(output_trec_run, 'w') as output:
            core.cascade.print_trec_run(output, states[-1]['preds'], y, qid, docno)
        logging.info('TREC run saved to %s' % output_trec_run)


def train(train_data, valid_data, costs, importance, n_stages, cutoffs, feature_partitions, alphas, **params):
    """Learn one ranker with SGD and L1 regularization.

    Args:
        n_stages: number of rankers in the cascade
        strategies: a dict of callback functions
    """
    x_train, y_train, qid_train, _ = train_data
    x_valid, y_valid, qid_valid, _ = valid_data

    running_costs = costs.copy()
    opened_features = np.array([], dtype=int)

    stages = []
    for i, features_to_open in enumerate(feature_partitions, 1):
        # retrieve the set of features open in this stage
        opened_features = np.union1d(opened_features, features_to_open)
        non_open_features = np.setdiff1d(np.arange(costs.shape[0]), opened_features)

        # hide non-open features in both training/validation sets
        x_train_prime = x_train.copy()
        x_train_prime[:, non_open_features] = 0

        x_valid_prime = None
        if x_valid is not None:
            x_valid_prime = x_valid.copy()
            x_valid_prime[:, non_open_features] = 0

        alpha = alphas.pop(0)

        print('stage %i: train (with alpha %f)' % (i, alpha))
        fit = _train((x_train_prime, y_train, qid_train),
                     (x_valid_prime, y_valid, qid_valid),
                     costs=running_costs, importance=importance,
                     alpha=alpha, **params)
        model = SGDClassifierModel(fit)

        cutoff = cutoffs.pop(0)
        prune = Prune(rank=cutoff) if cutoff else None
        stages.append((prune, model))

        # amend the cost (features used by the model are now free)
        used_features = np.flatnonzero(model.get_feature_mask())
        running_costs[used_features] = 0

    return stages


def _train(train_data, valid_data, costs, importance,
           max_iter=10, alpha=0.1, minibatch=1000, epochs=10,
           l1_ratio=1.0, penalty='none', eta0=0.01):
    """Train one cost-aware linear model using SGD.

    Args:
        max_iter: number of passes over the mini-batch (mini-epoch?)
        alpha: regularizer weight
        minibatch: size of a mini-batch
        epochs: number of passes over the training data
    """
    x_train, y_train, qid_train = train_data
    x_valid, y_valid, qid_valid = valid_data

    model = SGDClassifier(alpha=alpha,
                          verbose=False,
                          shuffle=False,
                          n_iter=max_iter,
                          learning_rate='constant',
                          penalty=penalty,
                          l1_ratio=l1_ratio,
                          eta0=eta0)

    model.classes_ = np.array([-1, 1])

    # fit SGD over the full data to initialize the model weights
    model.fit(x_train, y_train)

    valid_scores = (np.nan, np.nan, np.nan)
    if x_valid is not None:
        m = test_all(model.decision_function(x_valid), y_valid, qid_valid, 1)
        valid_scores = (m['ndcg@10'], m['p@10'], m['err@10'])
        print('[%3i]: weighted L1 %8.2f, cost %8d, features %4d, valid ndcg@10/p@10/err@10 %0.4f/%0.4f/%0.4f' %
              (0,
               np.sum(np.abs(model.coef_[0] * costs)),
               np.sum(costs[np.nonzero(model.coef_[0])]),
               np.count_nonzero(model.coef_[0]),
               valid_scores[0], valid_scores[1], valid_scores[2]))

    # SGD algorithm (Tsuruoka et al., 2009)
    u = np.zeros(x_train.shape[1])
    q = np.zeros(x_train.shape[1])

    for epoch in range(1, epochs + 1):
        for iterno, batch in enumerate(batch_generator(x_train, y_train, minibatch, x_train.shape[0]), 1):
            x, y = batch

            # call the internal method to specify custom classes, coef_init, and intercept_init
            model._partial_fit(x, y,
                               alpha=model.alpha,
                               C=1.0,
                               loss=model.loss,
                               learning_rate=model.learning_rate,
                               n_iter=1,
                               classes=model.classes_,
                               sample_weight=None,
                               coef_init=model.coef_,
                               intercept_init=model.intercept_)

            new_w = np.zeros(model.coef_.shape[1])
            u += model.eta0 * model.alpha * costs / float(x_train.shape[0])  # note the costs

            for i in range(len(model.coef_[0])):
                if model.coef_[0][i] > 0:
                    new_w[i] = max(0, model.coef_[0][i] - (u[i] + q[i]))
                elif model.coef_[0][i] < 0:
                    new_w[i] = min(0, model.coef_[0][i] + (u[i] - q[i]))
            q += new_w - model.coef_[0]
            model.coef_[0] = new_w

        valid_scores = (np.nan, np.nan, np.nan)
        if x_valid is not None:
            m = test_all(model.decision_function(x_valid), y_valid, qid_valid, 1)
            valid_scores = (m['ndcg@10'], m['p@10'], m['err@10'])
        print('[%3i]: weighted L1 %8.2f, cost %8d, features %4d, valid ndcg@10/p@10/err@10 %0.4f/%0.4f/%0.4f' %
              (epoch,
               np.sum(np.abs(model.coef_[0] * costs)),
               np.sum(costs[np.nonzero(model.coef_[0])]),
               np.count_nonzero(model.coef_[0]),
               valid_scores[0], valid_scores[1], valid_scores[2]))

    return model


def retrain(model_type, stages, train_data, valid_data,
            learning_rate, subsample, trees, nodes, up_to=0):
    params = {'max_depth': 7,
              'eta': learning_rate,
              'silent': True,
              'subsample': subsample}
    if model_type in ['GBDT']:
        params['objective'] = 'multi:softprob'
        params['eval_metric'] = 'mlogloss'
        score_function, set_classes = core.get_score_multiclass, True
    elif model_type in ['GBRT']:
        params['objective'] = 'reg:linear'
        params['eval_metric'] = 'rmse'
        score_function, set_classes = core.get_score, False
    elif model_type in ['LambdaMART']:
        params['objective'] = 'rank:pairwise'
        params['eval_metric'] = 'rmse'
        score_function, set_classes = core.get_score, False
    else:
        raise Exception()

    new_stages = []
    for i, (prune, model) in enumerate(stages, 1):
        used_features = np.flatnonzero(model.get_feature_mask())

        print('stage %i: retrain with %i features' % (i, used_features.size))

        _, y_train, _, _ = train_data
        class_weights = core.get_class_weights(y_train)

        import GBDT
        new_model = TreeModel(
            model=GBDT.train(train_data, valid_data, score_function, class_weights,
                             params, trees=trees, nodes=nodes,
                             set_classes=set_classes, features=used_features),
            score_function=score_function,
            class_weights=class_weights,
            n_features=train_data[0].shape[1])

        new_stages.append((prune, new_model))
        if i == up_to:
            break  # NOTE: stop as requested
    return new_stages


def retrain_with_RankSVM(stages, train_data, valid_data):
    new_stages = []
    for i, (prune, model) in enumerate(stages, 1):
        used_features = np.flatnonzero(model.get_feature_mask())

        print('stage %i: retrain with %i features' % (i, used_features.size))

        _, y_train, _, _ = train_data
        class_weights = core.get_class_weights(y_train)

        params = {'tol': float(1e-4),
                  'fit_intercept': True,
                  'cache_size': 20000,
                  'intercept_scaling': 1,
                  'class_weight': None,
                  'verbose': True,
                  'random_state': None,
                  'max_iter': 1000,
                  'loss': 'hinge',
                  'penalty': 'l2'}
        score_function = core.get_score

        import RankSVM
        new_model = SVMModel(
            model=RankSVM.train(train_data, valid_data, RankSVM.get_RankSVM, score_function,
                                class_weights, params, C=[0.1, 0.5, 1, 2], transform=True),
            score_function=score_function,
            class_weights=class_weights,
            n_features=train_data[0].shape[1])

        new_stages.append((prune, new_model))
    return new_stages


def train_disjoint_cascade(partition_criteria, train_file, validation_file, test_file,
                           costs_file=None, importance_file=None, model_prefix=None,
                           n_stages=3, cutoffs=[None, 10, 5], alpha=0.1, epochs=10, pairwise_transform=False,
                           GBDT_retraining=False):
    """Train a cascade over a partition of disjoint feature sets."""

    np.random.seed(0)  # freeze the randomness bit
    alphas = alpha if isinstance(alpha, list) else [alpha] * n_stages
    params = {'epochs': epochs,
              'l1_ratio': 1.0,
              'penalty': 'none'}

    scaler = MaxAbsScaler(copy=False)
    train_data, valid_data, test_data = load_data(
        train_file, validation_file, test_file, scaler=scaler)
    costs, importance = load_costs_data(
        costs_file, importance_file, n_features=train_data[0].shape[1])

    # these options don't go well together (or I haven't figured out how to make them)
    assert not (pairwise_transform and GBDT_retraining)

    # keep the original as GBDT won't work with polarized labels
    original_train_data = train_data

    # massage the data a bit ...
    x_train, y_train, qid_train, docno_train = train_data
    y_train = core.polarize(y_train)

    if pairwise_transform:
        from utils import per_query_transform_pairwise
        x_train, y_train = per_query_transform_pairwise(x_train.toarray(), y_train, qid_train)

    train_data = (x_train, y_train, qid_train, docno_train)

    is_qf = np.ones_like(costs)
    x = x_train.toarray()
    for j, _ in enumerate(costs):
        for a, b in group_offsets(qid_train):
            if (x[a:b, j] != x[a, j]).any():
                is_qf[j] = 0
                break

    # NOTE: costs has to be untainted (make copy before passing it to functions)
    partitions = partition_criteria(n_stages, is_qf, costs.copy(), importance)

    stages = train(train_data, valid_data, costs.copy(), importance, n_stages,
                   cutoffs=cutoffs, feature_partitions=partitions, alphas=alphas, **params)
    if GBDT_retraining:
        stages = retrain('GBDT', stages, original_train_data, valid_data,
                         trees=[5, 10, 50, 100, 500, 1000], nodes=[32])

    cascade = {'stages': stages,
               'scaler': scaler,
               'score_update': core.cascade.UpshiftUpdate(gap=0.1)}

    if model_prefix:
        save_model(cascade, model_prefix)
    predict(cascade, test_data, costs)


@baker.command(name='train')
def do_train(strategy, train_file, validation_file, test_file,
             costs_file=None, importance_file=None, model_prefix=None,
             n_stages=3, cutoffs="[None,10,5]", alpha="0.1", epochs=10,
             pairwise_transform=False, GBDT_retraining=False, use_query_features=False):
    """Train a disjoint cascade"""

    def no_partition(n_stages, is_qf, costs, _):
        return [np.arange(costs.shape[0])] * n_stages

    def random_partition(n_stages, is_qf, costs, _):
        if use_query_features:
            features = np.random.permutation(costs.shape[0])
        else:
            features = np.flatnonzero(1 - is_qf)
            np.random.shuffle(features)
        return np.array_split(features, n_stages)

    def cost_biased_partition(n_stages, is_qf, costs, _):
        if use_query_features:
            features = np.argsort(costs)
        else:
            nqf = np.flatnonzero(1 - is_qf)
            features = nqf[np.argsort(costs[nqf])]
        return np.array_split(features, n_stages)

    def importance_biased_partition(n_stages, is_qf, _, importance):
        if use_query_features:
            features = np.argsort(importance)
        else:
            nqf = np.flatnonzero(1 - is_qf)
            features = nqf[np.argsort(importance[nqf])]
        return np.array_split(features[::-1], n_stages)

    def efficiency_biased_partition(n_stages, is_qf, costs, importance):
        efficiency = importance / costs  # or any other return curves would do
        if use_query_features:
            features = np.argsort(efficiency)
        else:
            nqf = np.flatnonzero(1 - is_qf)
            features = nqf[np.argsort(efficiency[nqf])]
        return np.array_split(features[::-1], n_stages)  # in descending order

    if strategy in ['default', 'all']:
        partition = no_partition
    elif strategy in ['random']:
        partition = random_partition
    elif strategy in ['cost_biased', 'cost']:
        partition = cost_biased_partition
    elif strategy in ['importance_biased', 'importance']:
        partition = importance_biased_partition
    elif strategy in ['efficiency_biased', 'efficiency']:
        partition = efficiency_biased_partition
    else:
        print("Strategy not available: '%s'" % strategy)
        return

    train_disjoint_cascade(partition, train_file, validation_file, test_file,
                           costs_file, importance_file, model_prefix=model_prefix,
                           n_stages=n_stages, cutoffs=ast.literal_eval(cutoffs),
                           alpha=ast.literal_eval(alpha), epochs=epochs,
                           pairwise_transform=pairwise_transform, GBDT_retraining=GBDT_retraining)


@baker.command(name='train_budgeted_GBDT')
def do_train_budgeted_GBDT(train_file, validation_file, test_file, costs_file=None,
                           importance_file=None, model_prefix=None, budget=None,
                           trees='[5, 10, 50, 100, 500, 1000]', nodes='[32]'):
    """Train a 1-stage budgeted GBDT cascade"""

    train_data, valid_data, test_data = load_data(train_file, validation_file, test_file)
    costs, importance = load_costs_data(costs_file, importance_file,
                                        n_features=train_data[0].shape[1])

    x_train, _, _ = train_data
    x_train = x_train.toarray()

    # not all features will be used in a full model
    all_fids = [i for i in range(x_train.shape[1]) if any(x_train[:, i])]

    budget = float(budget)
    if budget:
        c = costs[all_fids]
        c[c.argsort()] = c[c.argsort()].cumsum()
        fids = [fid for fid, b in zip(all_fids, c) if b <= budget]
    else:
        fids = all_fids

    used_features = np.array(fids)
    # used_features = np.flatnonzero(model.get_feature_mask())

    print('Train a budgeted GBDT with %i features' % used_features.size)

    _, y_train, _ = train_data
    class_weights = core.get_class_weights(y_train)

    params = {'max_depth': 7,
              'eta': 0.1,
              'silent': True,
              'objective': 'multi:softprob',
              'eval_metric': 'mlogloss',
              'subsample': 0.5}

    import GBDT
    new_model = TreeModel(
        model=GBDT.train(train_data, valid_data, core.get_score_multiclass, class_weights,
                         params, trees=ast.literal_eval(trees), nodes=ast.literal_eval(nodes),
                         set_classes=True, features=used_features),
        score_function=core.get_score_multiclass,
        class_weights=class_weights,
        n_features=train_data[0].shape[1])

    cascade = {'stages': [(None, new_model)],
               'score_update': core.cascade.UpshiftUpdate(gap=0.1)}

    if model_prefix:
        save_model(cascade, model_prefix)
    predict(cascade, test_data, costs)


@baker.command(name='predict')
def do_predict(test_file, costs_file, model_file, output_trec_run=None, output_eval=None,
               override_cutoffs=None):
    """Run prediction with a saved cascade"""
    test_data = load_data_file(test_file)
    costs, _ = load_costs_data(costs_file, None, n_features=test_data[0].shape[1])

    cascade = load_model(model_file)
    if 'scaler' in cascade:
        cascade['scaler'].transform(test_data[0])

    if override_cutoffs:
        cutoffs = ast.literal_eval(override_cutoffs)
        logging.info('Override cutoffs with %s' % cutoffs)

        new_stages = []
        for i, (prune, model) in enumerate(cascade['stages']):
            new_stages.append((Prune(rank=cutoffs[i]), model))
        cascade['stages'] = new_stages

    predict(cascade, test_data, costs,
            output_trec_run=output_trec_run, output_eval=output_eval)


@baker.command(name='retrain')
def do_retrain(model_type, train_file, validation_file, model_file, new_model_file,
               test_file=None, costs_file=None, random=0, up_to=0,
               learning_rate="0.1", subsample="0.5", trees="[5,10,50,100,500,1000]", nodes="[32]",
               output_trec_run=None, output_eval=None):
    """Retrain a tree-based cascade using features learned in the linear models"""
    train_data = load_data_file(train_file)
    valid_data = (None,) * 4
    if validation_file:
        valid_data = load_data_file(validation_file)

    test_data = (None,) * 4
    costs = None
    if test_file is not None and costs_file is not None:
        test_data = load_data_file(test_file)
        costs, _ = load_costs_data(costs_file, None, n_features=test_data[0].shape[1])

    cascade = load_model(model_file)
    if 'scaler' in cascade:
        cascade['scaler'].transform(train_data[0])
        if valid_data[0] is not None:
            cascade['scaler'].transform(valid_data[0])
        if test_data[0] is not None:
            cascade['scaler'].transform(test_data[0])

    if random > 0:
        for _ in range(random):
            tree = 1 + np.random.randint(1000)
            node = np.random.choice([2, 4, 8, 16, 32, 64])
            print('tree %i, node %i' % (tree, node))
            new_cascade = cascade.copy()
            new_cascade['stages'] = retrain(model_type, cascade['stages'], train_data, valid_data,
                                            learning_rate=ast.literal_eval(learning_rate),
                                            subsample=ast.literal_eval(subsample),
                                            trees=[tree], nodes=[node], up_to=up_to)
            if test_data[0] is not None:
                predict(new_cascade, test_data, costs,
                        output_trec_run=output_trec_run, output_eval=output_eval)
        return

    cascade['stages'] = retrain(model_type, cascade['stages'], train_data, valid_data,
                                learning_rate=ast.literal_eval(learning_rate),
                                subsample=ast.literal_eval(subsample),
                                trees=ast.literal_eval(trees), nodes=ast.literal_eval(nodes), up_to=up_to)
    save_model(cascade, new_model_file)

    if test_data[0] is not None:
        predict(cascade, test_data, costs,
                output_trec_run=output_trec_run, output_eval=output_eval)


@baker.command(name='info')
def do_info(model_file):
    cascade = load_model(model_file)
    for i, (_, stage) in enumerate(cascade['stages'], 1):
        fids = np.flatnonzero(stage.get_feature_mask()) + 1
        print('stage', i)
        print('n_features', len(fids))
        print('fids', fids)


if __name__ == "__main__":
    logging.basicConfig(
        format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    baker.run()
