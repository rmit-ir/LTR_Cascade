from __future__ import print_function

import argparse
import itertools
import numpy as np
import sh


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-small', action='store_true',
                        help='run the search between 0 and 10, to the 2nd decimal place')
    parser.add_argument('-n', type=int,
                        help='number of stages (default: %(default)s)')
    parser.add_argument('-e', action='store_true',
                        help='add single-value rounds before sampling')
    parser.add_argument('n_configs', type=int, nargs='?')
    parser.set_defaults(n=3, n_configs=0)
    args = parser.parse_args()

    if args.small:
        alpha_values = [0.01, 0.03, 0.05, 0.08,
                        0.1, 0.3, 0.5, 0.8,
                        1, 3, 5, 8,
                        10, 30, 50, 80,
                        100, 300, 500, 800,
                        1000, 3000, 5000, 8000]
        pool = np.array(alpha_values, dtype=float)
    else:
        alpha_values = [1, 3, 5, 8,
                        10, 30, 50, 80,
                        100, 300, 500, 800,
                        1000, 3000, 5000, 8000,
                        10000, 30000, 50000, 8000,
                        100000, 300000, 500000, 800000,
                        1000000, 3000000, 5000000, 8000000,
                        10000000]
        pool = np.array(alpha_values, dtype=int)

    if args.e:
        for v in alpha_values:
            values = [v] * args.n
            print('[{}]'.format(','.join(map(str, values))))
    for i in range(args.n_configs):
        values = sorted(pool[np.random.randint(pool.size, size=args.n)], reverse=True)
        print('[{}]'.format(','.join(map(str, values))))
