from nltk.corpus import stopwords
import string
stop_words = set(stopwords.words('english'))
stop_words.update(set(string.punctuation))

import jsonlines
from transformers import AutoTokenizer
import numpy as np
add_prefix = False
tokenizer = AutoTokenizer.from_pretrained('t5-base')

import spacy
nlp = spacy.load('en_core_web_md')

def word_tokenize(text):
    doc = nlp(text)
    tokens = [token.text for token in doc]
    lemmas = [token.lemma_ for token in doc]
    return tokens, lemmas

def get_common_spans(s1, s2):

    len_s1 = len(s1)
    len_s2 = len(s2)
    record = [[0 for i in range(len_s2 + 1)] for j in range(len_s1 + 1)]

    maxNum = 0

    for i in range(len_s1):
        for j in range(len_s2):
            if s1[i] == s2[j]:
                record[i + 1][j + 1] = record[i][j] + 1
                if record[i + 1][j + 1] > maxNum:
                    maxNum = record[i + 1][j + 1]

    record = np.array(record)
    labels = ['O'] * len_s1
    while True:
        pos = record.argmax()
        i, j = pos // (len_s2 + 1), pos % (len_s2 + 1)
        maxNum = record[i][j]
        if maxNum > 0 and not all([True if x in stop_words else False for x in s1[i - maxNum: i]]):
            labels[i - maxNum: i] = ['I'] * maxNum
            record[:, j + 1 - maxNum: j + 1] = 0
        else:
            break

    return labels

def get_spans(s1, label_lists):
    labels = ['O'] * len(label_lists[0])
    for x in label_lists:
        for i, y in enumerate(x):
            if y == 'I':
                labels[i] = 'I'
    output = []
    cur_str = []
    for x, y in zip(s1, labels):
        if y == 'I':
            cur_str.append(x)
        elif cur_str:
            output.append(' '.join(cur_str))
            cur_str = []
    if cur_str:
        output.append(' '.join(cur_str))
    return output

def delete_enter(text):
    return ' '.join([x.strip() for x in text.split('\n')])

def get_model_input(data):
    data = list(data.values())[0]
    persona = f"[apprentice_persona] {delete_enter(data['apprentice_persona']).strip()}"
    dialog = []
    model_input = []
    tmp = []
    for idx, turn in enumerate(data['dialog_history']):
        action = turn['action'].strip()
        text = turn['text'].strip()
        if action == 'Apprentice => Wizard':
            role = 'Apprentice'
            dialog.append(f"[{role}] {text}")
        elif action == 'Wizard => Apprentice':
            role = 'Wizard'
            if tmp and add_prefix:
                prefix = f"[Search_Query] {', '.join(tmp)} "
                tmp = []
            else:
                prefix = ''
            dialog.append(prefix + f"[{role}] {text}")
        elif action == 'Wizard => SearchAgent':
            role = 'Search_Query'
            tmp.append(text)
        elif action == 'SearchAgent => Wizard':
            continue
        else:
            raise Exception('UNKNOWN ACTION')
        if role == 'Apprentice' and idx < len(data['dialog_history']) - 1:
            model_input.append({
                'dialogue': f"{persona} [dialog_history] {' '.join(dialog[max(0, len(dialog) - 8):])}".lower(),
                'query': []
            })
        elif role == 'Search_Query':
            if model_input:
                model_input[-1]['query'].append(text.lower())
            else:
                model_input.append({
                    'dialogue': f"{persona} [dialog_history] ".lower(),
                    'query': [text.lower()]
                })
        elif role == 'Wizard' and len(model_input) == 0:
            model_input.append({
                'dialogue': f"{persona} [dialog_history] ".lower(),
                'query': []
            })
    # postprocess
    final_output = []
    for x in model_input:
        context, lemmas = word_tokenize(x['dialogue'])
        queries = x['query']
        labels, mixed_queries = [], []
        for query in queries:
            if query != 'none':
                labels.append(get_common_spans(lemmas, word_tokenize(query)[1]))
        extracted_prefix = ''
        if labels:
            output = get_spans(context, labels)
            extracted_prefix = ', '.join(output)
        for query in queries:
            mixed_queries.append(' | '.join([extracted_prefix, query]))
        final_output.append({'dialogue': x['dialogue'], 'query': mixed_queries})
    return final_output

import os
if not os.path.exists('../../saved_data/woi_data_ext_gen_v2'):
    os.mkdir('../../saved_data/woi_data_ext_gen_v2')

for split in ['valid', 'test', 'train']:
    data = []
    max_len = 0
    with jsonlines.open(f'/cephfs/antewang/wizard_of_interent/{split}.jsonl', 'r') as reader:
        for line in reader:
            model_inputs = get_model_input(line)
            data += model_inputs
            max_len = max(max_len, len(tokenizer.tokenize(data[-1]['dialogue'])))
    print(len(data), max_len)
    # postprocess
    for x in data:
        if len(x['query']) == 0:
            x['query'].append('none')
    output_file = f'../../saved_data/woi_data_ext_gen_v2/{split}.json'
    with jsonlines.open(output_file, 'w') as writer:
        for x in data:
            writer.write(x)