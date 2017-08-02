import fileinput


if __name__ == '__main__':
    COST_UNKNOWN = 100000000

    for line in fileinput.input():
        for v in line.strip().split(','):
            v = float(v)
            print 10000 if v == COST_UNKNOWN else int(v)
