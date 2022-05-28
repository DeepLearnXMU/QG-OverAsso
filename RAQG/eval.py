# coding=utf-8
from collections import Counter

import jsonlines
from sacrebleu.metrics import BLEU
import numpy as np
from nltk.tokenize import word_tokenize

def distinct(seqs):
    """ Calculate intra/inter distinct 1/2. """
    batch_size = len(seqs)
    intra_dist1, intra_dist2 = [], []
    unigrams_all, bigrams_all = Counter(), Counter()
    for seq in seqs:
        unigrams = Counter(seq)
        bigrams = Counter(zip(seq, seq[1:]))
        intra_dist1.append((len(unigrams)+1e-12) / (len(seq)+1e-5))
        intra_dist2.append((len(bigrams)+1e-12) / (max(0, len(seq)-1)+1e-5))

        unigrams_all.update(unigrams)
        bigrams_all.update(bigrams)

    inter_dist1 = (len(unigrams_all)+1e-12) / (sum(unigrams_all.values())+1e-5)
    inter_dist2 = (len(bigrams_all)+1e-12) / (sum(bigrams_all.values())+1e-5)
    intra_dist1 = np.average(intra_dist1)
    intra_dist2 = np.average(intra_dist2)
    return intra_dist1, intra_dist2, inter_dist1, inter_dist2

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

ext_gen = True
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
    bleu1 = BLEU(max_ngram_order=1)
    bleu2 = BLEU(max_ngram_order=2)
    for step in range(50, 350, 50):
        data_path = '../saved_data/woi_data_np/valid.json'
        predict_path = f'../saved_data/RAQG/checkpoint-{step}/generated_predictions.txt'

        preds = []
        labels = []
        max_ref_num = 0
        with jsonlines.open(data_path, 'r') as reader:
            for line in reader:
                line['query'] = ext_gen_recover(line['query'])
                labels.append(line['query'])
                max_ref_num = max(max_ref_num, len(line['query']))

        with open(predict_path, 'r') as f:
            for i, line in enumerate(f.readlines()):
                preds.append(ext_gen_recover(line.strip()))

        r = 0
        decode_preds, decode_refs = [], []
        for pred, label in zip(preds, labels):
            if (label[0] == 'none' and pred == 'none') or (label[0] != 'none' and pred != 'none'):
                r += 1
            if label[0] != 'none':
                if pred == 'none':
                    pred = ''
                decode_preds.append(pred)
                decode_refs.append(label)

        bleu_labels = []
        for i in range(max_ref_num):
            bleu_labels.append([])
            for j in range(len(decode_refs)):
                if len(decode_refs[j]) > i:
                    bleu_labels[i].append(decode_refs[j][i])
                else:
                    bleu_labels[i].append(decode_refs[j][-1])

        print(f'----step {step}----')
        print(f'ACC : {r / len(labels)}')
        print(distinct([x.split() for x in decode_preds]))
        print(bleu1.corpus_score(decode_preds, bleu_labels))
        print(bleu2.corpus_score(decode_preds, bleu_labels))
        print(uni_F1_score(decode_preds, decode_refs))
