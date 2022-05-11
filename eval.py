# coding=utf-8
import jsonlines
from sacrebleu.metrics import BLEU
import numpy as np
from nltk.tokenize import word_tokenize

def uni_F1_score(preds, labels):
    f1_score = []
    for pred, label in zip(preds, labels):
        pred = word_tokenize(pred)
        _f1_score = 0
        for _label in label:
            _label = word_tokenize(_label)
            pred_len = len(pred)
            label_len = len(_label)
            common_len = len(set(pred) & set(_label))
            try:
                p, r = common_len / pred_len, common_len / label_len
            except:
                p, r = 0, 0
            if p == 0 or r == 0:
                _f1_score = max(_f1_score, 0)
            else:
                _f1_score = max(_f1_score, 2*p*r/(p+r))
        f1_score.append(_f1_score)
    return np.mean(f1_score)

ext_gen = False
cal_ext = False

def ext_gen_recover(x):
    if not ext_gen:
        return x
    try:
        if isinstance(x, str):
            if x == 'none':
                return x
            if cal_ext:
                y = ' '.join([_x.strip() for _x in x.split('|')[0].strip().split(',')])
                return y if y else 'none'
            else:
                return x.split('|')[1].strip()
        elif isinstance(x, list):
            return [_x.split('|')[1].strip() if _x != 'none' else _x for _x in x]
    except:
        return x

if __name__ == '__main__':
    bleu = BLEU()
    for step in range(1000, 10000, 1000):
    # for step in [10000]:
        data_path = '../../saved_data/woi_data_np/test.json'
        predict_path = f'../../saved_data/t5-v1_1-base-np/checkpoint-{step}/test_generated_predictions.txt'

        preds = []
        labels = []
        max_ref_num = 0
        with jsonlines.open(data_path, 'r') as reader:
            for line in reader:
                line['query'] = ext_gen_recover(line['query'])
                labels.append(line['query'])
                max_ref_num = max(max_ref_num, len(line['query']))

        bleu_labels = []
        for i in range(max_ref_num):
            bleu_labels.append([])
            for j in range(len(labels)):
                if len(labels[j]) > i:
                    bleu_labels[i].append(labels[j][i])
                else:
                    bleu_labels[i].append(labels[j][-1])

        with open(predict_path, 'r') as f:
            for i, line in enumerate(f.readlines()):
                preds.append(ext_gen_recover(line.strip()))

        r = 0
        for pred, label in zip(preds, labels):
            if (label[0] == 'none' and pred == 'none') or (label[0] != 'none' and pred != 'none'):
                r += 1

        print(f'ACC : {r/len(labels)}')
        print(bleu.corpus_score(preds, bleu_labels))
        print(uni_F1_score(preds, labels))