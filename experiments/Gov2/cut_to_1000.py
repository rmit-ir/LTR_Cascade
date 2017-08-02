from __future__ import print_function

import argparse
import collections
import fileinput


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--inplace', action='store_true', help='edit the run files inplace')
    parser.add_argument('files', metavar='file', nargs='*', help='input run files')
    args = parser.parse_args()

    count = collections.defaultdict(int)
    input_ = fileinput.input(args.files, inplace=args.inplace)
    for line in input_:
        if args.inplace and input_.isfirstline():
            count.clear()
        qid, _ = line.split(None, 1)
        if count[qid] < 1000:
            count[qid] += 1
            print(line, end='')
