from __future__ import print_function

import argparse
import itertools
import numpy as np
import sh


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', type=int,
                        help='number of stages (default: %(default)s)')
    parser.add_argument('n_configs', type=int, nargs='?')
    parser.set_defaults(n=3, n_configs=0)
    args = parser.parse_args()

    cutoff_values = range(20, 100, 10) + range(100, 1000, 100) + range(1000, 5000, 500)
    pool = np.array(cutoff_values, dtype=int)
    for i in range(args.n_configs):
        n_values = args.n - 1  # because the first cutoff is always 'None'
        values = sorted(np.random.choice(pool, n_values, replace=False), reverse=True)
        print('[None,{}]'.format(','.join(map(str, values))))
