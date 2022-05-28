# coding=utf-8
import jsonlines
from sacrebleu.metrics import BLEU
import numpy as np
from collections import Counter

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
        pred_len = len(pred)
        label_len = len(label)
        common_len = len(set(pred) & set(label))
        try:
            p, r = common_len / pred_len, common_len / label_len
        except:
            p, r = 0, 0
        if p == 0 or r == 0:
            f1_score.append(0)
        else:
            f1_score.append(2*p*r/(p+r))
    return np.mean(f1_score)

if __name__ == '__main__':
    bleu1 = BLEU(tokenize='zh', max_ngram_order=1)
    bleu2 = BLEU(tokenize='zh', max_ngram_order=2)
    for step in range(50, 350, 50):
        print(f'step {step}')
        data_path = '../saved_data/data/dev.json'
        predict_path = f'../saved_data/RAQG-zh/generated_predictions.txt'

        preds = []
        labels = []
        with jsonlines.open(data_path, 'r') as reader:
            for line in reader:
                ref = line['query']
                if ref == '无':
                    ref = '不检索'
                labels.append(ref)

        with open(predict_path, 'r') as f:
            for i, line in enumerate(f.readlines()):
                if ';' in line:
                    pred = line.strip().split(';')[1].strip()
                elif '|' in line:
                    pred = line.strip().split('|')[1].strip()
                else:
                    pred = line.strip()
                if pred.startswith('检索'):
                    pred = pred[2:]
                preds.append(pred)
                # if ';' in line:
                #     pred = line.strip().split(';')[0]
                #     pred = ''.join(pred.split(','))
                # else:
                #     pred = line.strip()
                # if pred.startswith('抽取'):
                #     pred = pred[2:]
                # preds.append(pred)

        r = 0
        decoded_preds, decoded_labels = [], []
        for pred, label in zip(preds, labels):
            if (label == '不检索' and pred == '不检索') or (label != '不检索' and pred != '不检索'):
                r += 1
            if label != '不检索':
                if pred == '不检索':
                    pred = ''
                decoded_preds.append(pred)
                decoded_labels.append(label)
        print(f'ACC : {r/len(labels)}')

        # decoded_preds, decoded_labels = preds, labels
        print(distinct(decoded_preds))
        print(bleu1.corpus_score(decoded_preds, [decoded_labels]))
        print(bleu2.corpus_score(decoded_preds, [decoded_labels]))
        print(uni_F1_score(decoded_preds, decoded_labels))