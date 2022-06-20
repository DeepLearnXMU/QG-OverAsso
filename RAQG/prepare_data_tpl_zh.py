# coding=utf-8
import os
import re
import jsonlines
from string import Template
from transformers import AutoTokenizer
import numpy as np

tokenizer = AutoTokenizer.from_pretrained('Langboat/mengzi-t5-base')

stopwords = set()
with open('../data/qianyan/cn_stopwords.txt', 'r') as f:
    for k in f.readlines():
        stopwords.add(k.strip())

def _uni_F1_score(pred, label):
    pred = detokenize(pred)
    label = detokenize(label)
    if pred is None:
        return -1
    if label is None:
        raise Exception(f'unformatted label {label}')
    if pred[0] == '不检索':
        if label[0] == '不检索':
            return 1.
        else:
            return 0.
    else:
        if label[0] == '不检索':
            return 0.
        else:
            f1_score = 0
            pred_len = len(pred)
            label_len = len(label)
            common_len = len(set(pred) & set(label))
            try:
                p, r = common_len / pred_len, common_len / label_len
            except:
                p, r = 0, 0
            if p == 0 or r == 0:
                f1_score = max(f1_score, 0)
            else:
                f1_score = max(f1_score, 2 * p * r / (p + r))
            return f1_score

def detokenize(x):
    raw_x = x
    if len(x) > 2 and x[1] == '不' and x[2] == '检索':
        return ['不检索']
    elif '|' in x:
        idx = x.index('|')
        x = x[idx + 1:]
        if len(x) > 2 and x[1] == '检索':
            return x[2:]
    return None

def uni_F1_score(pred, labels):
    return max([_uni_F1_score(pred, label) for label in labels])

def word_tokenize(text):
    text = re.sub('\s+', '', text)
    token_ids = tokenizer.tokenize(text)
    return token_ids

def clean_predictions(dialog, preds):
    # postprocess
    context_tokens = word_tokenize(dialog)
    queries = []
    for query in preds:
        if query != '不检索':
            if '|' not in query:
                continue
            query = query.split('|')[1].strip()
            query_tokens = word_tokenize(query)
            template = []
            is_stopwords = []
            for qt in query_tokens[2:]:
                if qt in context_tokens or qt in stopwords:
                    if qt in stopwords:
                        is_stopwords.append(1)
                    else:
                        is_stopwords.append(0)
                    template.append(qt)
                elif len(template) == 0 or template[-1] != '<extra_id_0>':
                    is_stopwords.append(2)
                    template.append('<extra_id_0>')
            # forward
            for i in range(1, len(is_stopwords)):
                if is_stopwords[i - 1] == 0 and is_stopwords[i] == 1:
                    is_stopwords[i] = 0
            # backward
            for i in list(range(0, len(is_stopwords) - 1))[::-1]:
                if is_stopwords[i + 1] == 0 and is_stopwords[i] == 1:
                    is_stopwords[i] = 0
            new_template = query_tokens[:2]
            for i in range(len(is_stopwords)):
                if is_stopwords[i] == 0:
                    new_template.append(template[i])
                elif is_stopwords[i] == 2 and (len(new_template) == 2 or new_template[-1] != '<extra_id_0>'):
                    new_template.append(template[i])
            queries.append(new_template + ['|'] + query_tokens)
        else:
            queries.append(word_tokenize(query))
    return queries


def mix_data(raw_data_file, predictions_file):
    with jsonlines.open(raw_data_file, 'r') as reader:
        raw_data = [line for line in reader]
    with open(predictions_file, 'r') as f:
        predictions = [line.strip() for line in f.readlines()]
    assert len(raw_data) * 10 == len(predictions)
    new_data = []
    cnt = 0
    for i in range(len(raw_data)):
        example = raw_data[i]
        dialog = example['dialogue']
        gold_refs = example['query']
        raw_preds = predictions[10 * i: 10 * (i + 1)]
        preds = clean_predictions(dialog, raw_preds)
        scores = [uni_F1_score(pred, gold_refs) for pred in preds]
        new_preds, new_scores = [], []
        for pred, score in zip(preds, scores):
            if score > -1:
                new_preds.append(pred)
                new_scores.append(score)
            else:
                cnt += 1
        new_example = {'dialogue': dialog, 'query': gold_refs, 'candidate': new_preds, 'score': new_scores}
        new_data.append(new_example)
    print(cnt)
    return new_data


if __name__ == '__main__':
    data = []
    for i in [0, 1, 2, 3, 4]:
        data += mix_data(f'../data_zh_tpl_gen/train_{i}_.json',
                         f'../mengzi-t5-base-tpl-gen-{i}f/bm_{i}_generated_predictions.txt')
    with jsonlines.open('train_tpl_zh.json', 'w') as writer:
        for line in data:
            writer.write(line)

