from __future__ import print_function

import baker
import logging

import core.io
from core.cascade import group_offsets


def truncate_data(x, y, qid, docno, k):
    """Truncate each ranked list down to at most k documents"""
    import numpy as np

    idx = np.concatenate([np.arange(a, min(a + k, b)) for a, b in group_offsets(qid)])
    new_docno = docno[idx] if docno is not None else None
    return x[idx], y[idx], qid[idx], new_docno


@baker.command
def make_npz(input_file, npz_file, k=0):
    """Convert input data (SVMLight or .npz) into .npz format"""
    if input_file.endswith('.npz'):
        x, y, qid, docno = core.io.load_npz(input_file)
    else:
        x, y, qid, docno = core.io.load_svmlight_file(input_file)

    # eliminate explicit zeros
    x.eliminate_zeros()

    # truncate data as necessary
    if k:
        x, y, qid, docno = truncate_data(x, y, qid, docno, k)
    core.io.save_npz(npz_file, x, y, qid, docno)


@baker.command
def merge_npz(*npz_file):
    """Merge multiple npz files (*** EXPERIMENTAL ***)"""
    import numpy as np
    import scipy.sparse

    dest_npz_file = npz_file[-1]  # the last filename is the destination
    npz_file = npz_file[:-1]

    x_list, y_list, qid_list, docno_list = [], [], [], []
    for fname in npz_file:
        x, y, qid, docno = core.io.load_npz(fname)
        x.eliminate_zeros()  # eliminate explicit zeros
        print(fname, x.shape, y.shape, qid.shape, docno.shape,
              'fid:[{}, {}]'.format(x.indices.min(), x.indices.max()))
        x_list.append(x)
        y_list.append(y)
        qid_list.append(qid)
        docno_list.append(docno)

    n_features = max(x.shape[1] for x in x_list)
    for i in range(len(x_list)):
        if x_list[i].shape[1] == n_features:
            continue
        new_shape = (x_list[i].shape[0], n_features)
        x_list[i] = scipy.sparse.csr_matrix((x_list[i].data, x_list[i].indices, x_list[i].indptr),
                                            shape=new_shape)

    x_new = scipy.sparse.vstack(x_list)
    print('x', type(x_new), x_new.shape, 'fid:[{}, {}]'.format(x_new.indices.min(), x_new.indices.max()))
    y_new = np.concatenate(y_list)
    print('y', type(y_new), y_new.shape)
    qid_new = np.concatenate(qid_list)
    print('qid', type(qid_new), qid_new.shape)
    docno_new = np.concatenate(docno_list)
    print('docno', type(docno_new), docno_new.shape)

    core.io.save_npz(dest_npz_file, x_new, y_new, qid_new, docno_new)


@baker.command
def show_npz_info(*npz_file):
    import numpy as np

    for fname in npz_file:
        print('filename', fname)
        x, y, qid, docno = core.io.load_npz(fname)
        if docno is not None:
            print('x', x.shape, 'y', y.shape, 'qid', qid.shape, 'docno', docno.shape)
        else:
            print('x', x.shape, 'y', y.shape, 'qid', qid.shape, 'docno', None)

        print('labels:', {int(k): v for k, v in zip(*map(list, np.unique(y, return_counts=True)))})

        unique_qid = np.unique(qid)
        print('qid (unique):', unique_qid.size)
        print(unique_qid)
        print()


@baker.command
def make_qrels(data_file, qrels_file):
    """Create qrels from an svmlight or npz file."""
    with open(qrels_file, 'wb') as out:
        if data_file.endswith('.npz'):
            _, y, qid, docno = core.io.load_npz(data_file)
            for a, b in group_offsets(qid):
                if docno is None:
                    docno_string = ['%s.%d' % (qid[a], i) for i in range(1, b - a + 1)]
                else:
                    docno_string = docno[a:b]
                for d, rel in zip(docno_string, y[a:b]):
                    out.write('%s 0 %s %d\n' % (qid[a], d, rel))
        else:
            for qid, docno, rel in core.io.parse_svmlight_into_qrels(data_file):
                out.write('%s 0 %s %d\n' % (qid, docno, rel))


@baker.command
def make_svm(*csv_file):
    """Convert CSV files into SVMLight format

    Format: <label>,<query id>,<docno>,f1,f2,...,fn
    """
    import itertools
    import pandas as pd

    fid = itertools.count(1)
    frames = []
    for fname in csv_file:
        df = pd.read_csv(fname, sep=',', header=None)
        names = (['rel', 'qid', 'docno'] +
                 ['f{}'.format(next(fid)) for _ in range(df.columns.size - 3)])
        df.columns = names
        frames.append(df)

    fid_end = next(fid)

    fields = ['f{}'.format(i) for i in range(1, fid_end)]
    fids = ['{}'.format(i) for i in range(1, fid_end)]

    # merge data frames
    df_all = reduce(lambda l, r: pd.merge(l, r, how='inner', on=['qid', 'docno']), frames)
    df_all['rel'] = df_all['rel_x'].astype(int)
    df_all['qid'] = df_all['qid'].astype(str)
    df_all['docno'] = df_all['docno'].astype(str)
    print(df_all.head())

    for index, row in df_all.iterrows():
        vector = ' '.join(['{}:{:.6f}'.format(k, v) for k, v in zip(fids, row[fields])])
        print('{rel} qid:{qid} {vector} # {docno}'.format(
            rel=row['rel_x'], qid=row['qid'], vector=vector, docno=row['docno']))


@baker.command
def make_run(data_file, scores_file, generate_docno=True):
    """Create run file from an svmlight/npz file and a scores file"""
    scores = core.io.load_scores(scores_file)
    if data_file.endswith('.npz'):
        _, y, qid = core.io.load_npz(data_file)
        for a, b in group_offsets(qid):
            for i in range(1, b - a + 1):
                docno = '%s.%d' % (qid[a], i)
                print('%s Q0 %s 0 %f %s' % (qid[a], docno, scores[a + i - 1], 'eval.py'))
    else:
        qrels = core.io.parse_svmlight_into_qrels(data_file, generate_docno=generate_docno)
        for (qid, docno, _), score in zip(qrels, scores):
            print('%s Q0 %s 0 %f %s' % (qid, docno, score, 'eval.py'))


if __name__ == "__main__":
    logging.basicConfig(
        format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    baker.run()
