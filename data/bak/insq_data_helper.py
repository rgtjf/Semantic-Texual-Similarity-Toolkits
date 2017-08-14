# coding: utf8
from __future__ import print_function
from __future__ import unicode_literals

import sys
sys.path.append('..')

import codecs
import json
import six
from stst import utils


test_file = 'multi_test.tsv'
insq_word_file = 'ins_test_mulwd.tsv'
insq_char_file = 'ins_test_mulch.tsv'


def process(sent, max_sent_len):
    """
    sent: list
    """
    if type(sent) == six.text_type:
        sent = sent.split()

    if len(sent) > max_sent_len:
        sent = sent[:max_sent_len]

    for i in range(len(sent), max_sent_len):
        sent.append('<a>')

    sent = '_'.join(sent)

    return sent


def get_word():
    fw = codecs.open(insq_word_file, 'w', encoding='utf8')

    for line in codecs.open(test_file, encoding='utf8'):
        # 1	yunqi	791	随行 人员 需要 报名 吗 ？	338	本人 已经 报名 ， 如 有 随行 人员 是否 需要 单独 报名 ？ 或者 直接 前往 现场 报名 签到 即 可 ？
        items = line.strip().split('\t')
        label = items[0]
        qqid = items[2]
        sa = process(items[3], 100)
        sb = process(items[5], 100)
        print('{} {} {} {}'.format(label, qqid, sa, sb), file=fw)

get_word()

fw = codecs.open(insq_char_file, 'w', encoding='utf8')
for line in codecs.open(test_file, encoding='utf8'):
    # 1	yunqi	791	随行 人员 需要 报名 吗 ？	338	本人 已经 报名 ， 如 有 随行 人员 是否 需要 单独 报名 ？ 或者 直接 前往 现场 报名 签到 即 可 ？
    items = line.strip().split('\t')
    label = items[0]
    qqid  = items[2]
    sa    = utils.word2char(items[3])
    sb    = utils.word2char(items[5])
    sa    = process(sa, 100)
    sb    = process(sb, 100)
    print('{} {} {} {}'.format(label, qqid, sa, sb), file=fw)