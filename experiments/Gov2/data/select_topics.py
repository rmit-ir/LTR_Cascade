from __future__ import print_function

import argparse
import fileinput

from tqdm import tqdm


def get_topics(spec, from_file=False):
    result = []
    if from_file:
        result.extend(map(str, [int(l.strip()) for l in open(spec)]))
    else:
        for term in spec.strip().split(','):
            if '-' in term:
                a, b = map(int, term.split('-'))
                result.extend(map(str, range(a, b + 1)))
            else:
                result.append(term)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--from-file', action='store_true', help='load topics from file')
    parser.add_argument('topic_spec', help='topic ranges or file input (with --from-file)')
    parser.add_argument('input_files', metavar='input', nargs='+', help='input SVMLight files')
    args = parser.parse_args()

    selected_topics = get_topics(args.topic_spec, from_file=args.from_file)
    buf = {t: [] for t in selected_topics}
    for line in tqdm(fileinput.input(args.input_files), desc='read the input topics'):
        if line.startswith('#'):
            continue
        _, qid, _ = line.split(None, 2)

        assert qid.startswith('qid:')
        if qid[4:] in buf:
            buf[qid[4:]].append(line)

    for topic in tqdm(selected_topics, desc='write to the output'):
        for line in buf[topic]:
            print(line, end='')
