# coding: utf8
from __future__ import print_function
import config, data_tools


# f = utils.create_write_file('../resources/data/sts-en-en/input.all2.txt')
# data = utils.create_read_file('../resources/data/sts-en-en/input.all.txt').readlines()
# gs = utils.create_read_file('../resources/data/sts-en-en/gs.all.txt').readlines()
# for sent, sc in zip(data, gs):
#     sent = sent.strip()
#     sc = sc.strip()
#     print(sent, sc)
#     print(sent + '\t' + sc, file=f)
print(len('asdf'))
print('%s'%True)
s = [[u'a', '#'], [u'cat', 'n'], [u'standing', 'n'], [u'on', '#'], [u'tree', 'n'], [u'branch', 'n'], [u'.', '#']]

def f():
    x  =[ 1, 2, 3]
    y = x
    y[0] = 2
    print(x)
    return [1], [2,3]

print(zip(f(), f()))
ss = []
ss = [s, s]
for idx, [lemma, pos] in enumerate(s):
    print(idx, lemma, pos)