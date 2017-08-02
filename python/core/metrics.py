import math
import numpy as np


# DCG
def dcg(r, k=None):
    """The Burges et al. (2005) version of DCG.

    This is what everyone uses (except trec_eval)
    :param r: results
    :param k: cut-off
    :return: sum (2^ y_i - 1) / log (i +2)
    """
    result = sum([(pow(2, rel) - 1) / math.log(rank + 2, 2) for rank, rel in enumerate(r[:k])])
    return result


def dcg_trec(r, k=None):
    """The `trec_eval` version of DCG

    :param r: results
    :param k: cut-off
    :return: sum rel_i / log(i + 2)
    """
    result = sum([rel / math.log(rank + 2, 2) for rank, rel in enumerate(r[:k])])
    return result


# nDCG
def ndcg(x, k=None, dcg=dcg):
    """
    Args:
        k: cutoff
        dcg_func: set to `dcg` or `dcg_trec`
    """
    ideal = dcg(sorted(x, reverse=True), k)
    if ideal == 0:
        return 0
    result = dcg(x, k) / ideal
    return result


def err(x, k, g_max):
    res = 0
    p = 1
    norm = pow(2, g_max)
    for i, g in enumerate(x[:k], 1):
        R = float(pow(2, g) - 1) / norm
        res += p * R / i
        p *= (1 - R)
    return res


def patk(labels, k):
    """We assume an array with 0's and 1's
        If there are no relevant documents (all zeroes) will return zero
    """
    return (0. + np.sum(labels[0:k])) / k


def mean_ap(labels, k=None, n_rel=None):
    """We assume an array with 0's and 1's
        If there are no relevant documents (all zeroes) will return zero
    """
    if n_rel is None:
        n_rel = np.sum(labels[0:k])
    if n_rel == 0:
        return 0

    idx = np.nonzero(labels[0:k])[0]
    score = np.sum((np.arange(idx.size) + 1.0) / (idx + 1.0))
    return score / n_rel


def print_trec_eval(groups):
    resfile = open('run.res', 'w')
    relfile = open('run.rel', 'w')

    qid = 0
    for s, l in groups:
        qid += 1
        new_l = [x for (y, x) in sorted(zip(s, l), key=lambda pair: pair[0], reverse=True)]
        new_s = sorted(s, reverse=True)
        for j in range(len(new_s)):
            resfile.write("Q%i Q0 Q%iD%i %i %f RUN\n" % (qid, qid, j+1, j+1, new_s[j]))
            relfile.write("Q%i 0 Q%iD%i %i\n" % (qid, qid, j+1, new_l[j]))
    relfile.close()
    resfile.close()


def ndcg_one(scores, labels):
    a = [x for (y, x) in sorted(zip(scores, labels), key=lambda pair: pair[0], reverse=True)]
    return ndcg(a, len(a))


def test_ndcg(scores, labels, query_ids, average=True):
    labels = np.maximum(labels, 0)
    groups, _ = group_per_query(scores, labels, query_ids)
    ndcg_scores = []
    for s, l in groups:
        idx = np.argsort(s)
        a = l[idx[::-1]]
        ndcg_scores.append(ndcg(a, len(a)))
    if average:
        return sum(ndcg_scores) / len(ndcg_scores)
    else:
        return ndcg_scores


def test_map(scores, labels, query_ids, average=True):
    labels = np.maximum(labels, 0)
    groups, _ = group_per_query(scores, labels, query_ids)
    map_scores = []
    for s, l in groups:
        idx = np.argsort(s)
        a = l[idx[::-1]]
        binary = [int(v > 0) for v in a]
        map_scores.append(mean_ap(binary))
    if average:
        return sum(map_scores) / len(map_scores)
    else:
        return map_scores


def group_per_query(pred, y, query_groups):
    # offsets, qids = group_offsets(query_groups, return_values=True)
    # return [(pred[a:b], y[a:b]) for a, b in offsets], qids

    qid = query_groups[0]
    ly = list()
    lpred = list()
    groups = list()
    unique_ids = [qid]

    for i in range(len(query_groups)):
        if qid != query_groups[i]:  # new query
            yarray = np.array(ly)
            predarray = np.array(lpred)
            groups.append((predarray, yarray))
            qid = query_groups[i]
            unique_ids.append(qid)
            ly = list()
            lpred = list()
        lpred.append(pred[i])
        ly.append(y[i])
    #groups.append((predarray, yarray))
    groups.append((np.array(lpred), np.array(ly)))
    return groups, unique_ids


def test_all(scores, labels, query_ids, rel=0, relcount={}, average=True):
    from collections import defaultdict
    max_grade = int(np.max(labels))
    labels = np.maximum(labels, 0)
    groups, qids = group_per_query(scores, labels, query_ids)
    # metrics = {"map": 0, "ndcg@5": 0, "ndcg@10": 0, "ndcg@20": 0, "dcg@5": 0, "dcg@10": 0,
    #            "dcg@20": 0, "err": 0,
    #            "p@1": 0, "p@2": 0, "p@5": 0, "p@10": 0, "p@20": 0, "dcg": 0, "ndcg": 0}

    if average:
        metrics = defaultdict(float)
    else:
        metrics = {k: [] for k in ('p@1', 'p@2', 'p@5', 'p@10', 'p@20', 'err@5', 'err@10', 'err@20',
                                   'dcg@5', 'dcg@10', 'dcg@20', 'ndcg@5', 'ndcg@10', 'ndcg@20',
                                   'map', 'err', 'dcg', 'ndcg')}

    # for (s, l) in groups:
    #     # sorts x using y as a key
    #     # a = [x for (y, x) in sorted(zip(s, l), key=lambda pair: pair[0], reverse=True)]
    #     idx = np.argsort(s)
    #     a = l[idx[::-1]]
    #     # print("scores %s labels %s array %s" %(s,l,a))
    #     # binary = map(lambda x: 1 if x > rel  else  0, a)
    #     binary = [int(v > 0) for v in a]

        # shallow metrics
    if average:
        for (s, l), qid in zip(groups, qids):
            idx = np.argsort(s)
            a = l[idx[::-1]]
            binary = [int(v > 0) for v in a]

            metrics["p@1"] += patk(binary, 1)
            metrics["p@2"] += patk(binary, 2)
            metrics["p@5"] += patk(binary, 5)
            metrics["p@10"] += patk(binary, 10)
            metrics["p@20"] += patk(binary, 20)
            metrics["err@5"] += err(a, 5, max_grade)
            metrics["err@10"] += err(a, 10, max_grade)
            metrics["err@20"] += err(a, 20, max_grade)
            metrics["dcg@5"] += dcg(a, 5)
            metrics["dcg@10"] += dcg(a, 10)
            metrics["dcg@20"] += dcg(a, 20)
            metrics["ndcg@5"] += ndcg(a, 5)
            metrics["ndcg@10"] += ndcg(a, 10)
            metrics["ndcg@20"] += ndcg(a, 20)

            # deep metrics
            metrics["map"] += mean_ap(binary, n_rel=relcount.get(qid, None))
            metrics["err"] += err(a, None, max_grade)
            metrics["dcg"] += dcg(a)
            metrics["ndcg"] += ndcg(a)

        for k in metrics.keys():
            metrics[k] /= len(groups) + 0.
        return metrics
    else:
        for (s, l), qid in zip(groups, qids):
            idx = np.argsort(s)
            a = l[idx[::-1]]
            binary = [int(v > 0) for v in a]

            metrics["p@1"].append(patk(binary, 1))
            metrics["p@2"].append(patk(binary, 2))
            metrics["p@5"].append(patk(binary, 5))
            metrics["p@10"].append(patk(binary, 10))
            metrics["p@20"].append(patk(binary, 20))
            metrics["err@5"].append(err(a, 5, max_grade))
            metrics["err@10"].append(err(a, 10, max_grade))
            metrics["err@20"].append(err(a, 20, max_grade))
            metrics["dcg@5"].append(dcg(a, 5))
            metrics["dcg@10"].append(dcg(a, 10))
            metrics["dcg@20"].append(dcg(a, 20))
            metrics["ndcg@5"].append(ndcg(a, 5))
            metrics["ndcg@10"].append(ndcg(a, 10))
            metrics["ndcg@20"].append(ndcg(a, 20))

            # deep metrics
            metrics["map"].append(mean_ap(binary, n_rel=relcount.get(qid, None)))
            metrics["err"].append(err(a, None, max_grade))
            metrics["dcg"].append(dcg(a))
            metrics["ndcg"].append(ndcg(a))

        return metrics
