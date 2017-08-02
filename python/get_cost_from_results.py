from __future__ import print_function

import argparse
import csv
import numpy as np
import os.path
import re
import sys


def get_metric_from_results(fname, metric):
    for line in open(fname):
        k, v = line.split()
        if k == metric:
            return v


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-tab', action='store_true', help='use tab as the separator')
    parser.add_argument('files', metavar='file', nargs='*')
    args = parser.parse_args()

    tags = {}
    for fname in args.files:
        tag = re.sub(r'\.f\d\.', '.all.', os.path.basename(fname))
        if tag not in tags:
            tags[tag] = []
        tags[tag].append(fname)

    header = ['type', 'data', 'method', 'alphas', 'cutoffs', 'ndcg', 'cost', 'n_stages', 'run']
    if args.tab:
        writer = csv.writer(sys.stdout, delimiter='\t')
    else:
        writer = csv.writer(sys.stdout)
    writer.writerow(header)

    for tag, fnames in sorted(tags.items()):
        m = re.match(r'(\S+?)\.(\S+?)\.(\S+?)\.all.alpha=(\[.*?\]).cutoffs=(\[.*?\])', tag)
        if m:
            settype, data, method, alpha, cutoffs = m.groups()
            cost = int(np.average([float(get_metric_from_results(name, 'cascade_cost_per_doc'))
                                   for name in fnames]))
            ndcg = float(np.average([float(get_metric_from_results(name, 'cascade_ndcg'))
                                     for name in fnames]))
            run = tag.replace(settype, 'run')
            row = {'type': settype, 'data': data, 'method': method,
                   'alphas': alpha, 'cutoffs': cutoffs, 'cost': cost, 'ndcg': ndcg,
                   'n_stages': cutoffs.count(',') + 1, 'run': run}
            writer.writerow([row[k] for k in header])
