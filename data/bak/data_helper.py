# coding: utf8
from __future__ import print_function
from __future__ import unicode_literals

import codecs
import json

all_data_file = './all-data.format.txt'
segmented_data_file = './yunxi-all-data.format.segmented.tsv'
yunxi_json_file = 'yunxi.json'

test_file = 'test.tsv'
standard_file = 'standard_yunxi.tsv'
test_multi_file = 'multi_test.tsv'


def generate_json_file():
    data = []
    for line in codecs.open(segmented_data_file, encoding='utf8'):
        items = line.strip().split('\t')
        question_type = items[0]
        standard_question = items[1]
        expanded_question = items[2]
        data.append([question_type, standard_question, expanded_question])

    index = 0
    q_id = 1
    with codecs.open(yunxi_json_file, 'w', encoding='utf8') as fw:
        for line in codecs.open(all_data_file, encoding='utf8'):
            items = line.strip().split(chr(3))
            q_type = items[0]
            expanded_question_list = items[2].split(chr(1))

            standard_wd = data[index][1]
            cands_wd = []
            for i in range(len(expanded_question_list)):
                cands_wd.append(data[index][2])
                index += 1
            obj = {'id': q_id, 'type': q_type,
                   'standard_ch': items[1],
                   'candidates_ch': expanded_question_list,
                   'answer_ch': items[3],
                   'standard_wd': standard_wd,
                   'candidates_wd': cands_wd,
                   }
            obj_str = json.dumps(obj, ensure_ascii=False)
            q_id += 1
            print(obj_str, file=fw)
generate_json_file()

""" generate standard file """
standard_list = []
fs = codecs.open(standard_file, 'w', encoding='utf8')
for line in codecs.open(yunxi_json_file, encoding='utf8'):
    obj = json.loads(line)
    if obj['type'] == 'yunqi':
        standard_list.append(obj)
        print('{}\t{}\t{}'.format(obj['id'], obj['type'], obj['standard_wd']), file=fs)


""" generate test file """
fw = codecs.open(test_file, 'w', encoding='utf8')
index = 1
for line in codecs.open(yunxi_json_file, encoding='utf8'):
    obj = json.loads(line)
    # print(obj['id'], obj['candidates_wd'])
    for item in obj['candidates_wd']:
        # print('{}\t{}\t{}'.format(obj['type'], obj['id'], item), file=fw)
        for std in standard_list:
            label = 1 if std['id'] == obj['id'] else 0
            print('{}\t{}\t{}\t{}\t{}\t{}'.format(
                label, obj['type'], index, item,
                std['id'], std['standard_wd']), file=fw)
        index += 1

fw.close()

fw = codecs.open(test_multi_file, 'w', encoding='utf8')
index = 1
for line in codecs.open(yunxi_json_file, encoding='utf8'):
    obj = json.loads(line)
    # print(obj['id'], obj['candidates_wd'])
    for item in obj['candidates_wd']:
        # print('{}\t{}\t{}'.format(obj['type'], obj['id'], item), file=fw)
        for std in standard_list:
            label = 1 if std['id'] == obj['id'] else 0
            print('{}\t{}\t{}\t{}\t{}\t{}'.format(
                label, obj['type'], index, item,
                std['id'], std['standard_wd']), file=fw)
        index += 1

fw.close()
