import itertools
import numpy as np
import scipy

import sklearn.datasets


def load_svmlight_file(svmlight_file):
    """Load data from svmlight file"""
    x, y, qid = sklearn.datasets.load_svmlight_file(svmlight_file, query_id=True)
    docno = []
    for line in open(svmlight_file):
        pos = line.find('#')
        if pos == 0:
            continue
        elif pos > 0:
            text = line[pos+1:].strip()
            if text.startswith('docno:'):
                text = text[6:]
            docno.append(text)
    if len(docno) == qid.size:
        docno = np.array(docno, dtype=str)
    else:
        docno = None
    return x, y, qid, docno


def save_npz(filename, x, y, qid, docno, compressed=True):
    """Save svmlight data to a file in .npz format."""
    saver = np.savez_compressed if compressed else np.savez

    data = {'x_data': x.data,
            'x_indices': x.indices,
            'x_indptr': x.indptr,
            'x_shape': x.shape,
            'y': y,
            'qid': qid}
    if docno is not None:
        data['docno'] = docno
    saver(filename, **data)


def load_npz(filename):
    """Load svmlight data from a .npz file."""
    loader = np.load(filename)
    x = scipy.sparse.csr_matrix((loader['x_data'], loader['x_indices'], loader['x_indptr']),
                                shape=loader['x_shape'])
    y = loader['y']
    qid = loader['qid']
    docno = None
    try:
        docno = loader['docno']  # for backward compatibility
    except KeyError:
        pass

    return x, y, qid, docno


def load_scores(filename):
    """Load scores from text file"""
    return np.loadtxt(filename)


def _parse_svmlight_into_qrels(filename):
    """Yield a sequence of qrel rows from svmlight data."""
    for line in file(filename):
        if line.startswith('#'):
            continue
        if '#' in line:
            head, comment = [part.strip() for part in line.split('#', 1)]
        else:
            head, comment = line.strip(), None
        fields = head.split()
        assert fields[1].startswith('qid:')
        qid = fields[1][4:]  # qid:XXXX
        rel = int(fields[0])
        docno = comment.split()[0] if comment else None
        yield qid, docno, rel


def parse_svmlight_into_qrels(filename, generate_docno=True):
    """Yield a sequence of qrel rows from svmlight data."""
    for k, grp in itertools.groupby(_parse_svmlight_into_qrels(filename), lambda x: x[0]):
        if generate_docno:
            for i, (qid, _, rel) in enumerate(grp, 1):
                docno = '%s.%d' % (qid, i)
                yield qid, docno, rel
        else:
            for qid, docno, rel in grp:
                yield qid, docno, rel
