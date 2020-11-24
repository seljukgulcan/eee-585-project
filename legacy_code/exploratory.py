from collections import namedtuple
import csv

Nonzero = namedtuple('Nonzero', ['row', 'col', 'rating'])


def iter_nnz_file(filename):
    with open(filename) as file:
        reader = csv.reader(file)
        next(reader)  # skip the header
        count = 0
        for row in reader:
            count += 1
            if count % 400_000 == 0:
                print(count)
            yield (Nonzero(int(row[0]), int(row[1]), float(row[2])))

nnz_list = []
filename = None

with open(filename) as file:
    reader = csv.reader(file)
    next(reader) # skip the header
    count = 0
    for row in reader:
        count += 1
        if count % 400_000 == 0:
            print(count)
        nnz_list.append(Nonzero(int(row[0]), int(row[1]), float(row[2])))

