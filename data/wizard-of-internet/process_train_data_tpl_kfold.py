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
        context_tokens, context_lemmas = word_tokenize(x['dialogue'])
        queries = x['query']
        mixed_queries = set()
        for query in queries:
            query_tokens, query_lemmas = word_tokenize(query)
            if query != 'none':
                template = []
                is_stopwords = []
                for qt, ql in zip(query_tokens, query_lemmas):
                    if ql in context_lemmas or qt in context_lemmas or qt in stop_words:
                        if qt in stop_words:
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
                for i in list(range(0, len(is_stopwords) -1))[::-1]:
                    if is_stopwords[i + 1] == 0 and is_stopwords[i] == 1:
                        is_stopwords[i] = 0
                new_template = []
                for i in range(len(is_stopwords)):
                    if is_stopwords[i] == 0:
                        new_template.append(template[i])
                    elif is_stopwords[i] == 2 and (len(new_template) == 0 or new_template[-1] != '<extra_id_0>'):
                        new_template.append(template[i])
                mixed_queries.add(f"{' '.join(new_template)} | {query}")
        final_output.append({'dialogue': x['dialogue'], 'query': list(mixed_queries)})
    return final_output

k_fold = 3
output_dir = f'../../saved_data/data_en_tpl_gen'

import os
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

for split in ['valid', 'test', 'train']:
    data = []
    max_len = 0
    with jsonlines.open(f'../../saved_data/wizard_of_interent/{split}.jsonl', 'r') as reader:
        for line in reader:
            model_inputs = get_model_input(line)
            data += model_inputs
            max_len = max(max_len, len(tokenizer.tokenize(data[-1]['dialogue'])))
    print(len(data), max_len)

    # postprocess
    for x in data:
        if len(x['query']) == 0:
            x['query'].append('none')
    
    with jsonlines.open(output_file, 'w') as writer:
        for x in data:
            writer.write(x)
    
    if split == 'train':
        for i in range(k_fold):
            with jsonlines.open(f'{output_dir}/{split}_{i}.json', 'w') as writer:
                for j in range(len(data)):
                    if j % k_fold != i:
                        writer.write(data[j])
            with jsonlines.open(f'{output_dir}/{split}_{i}_.json', 'w') as writer:
                for j in range(len(data)):
                    if j % k_fold == i:
                        writer.write(data[j])
    output_file = f'{output_dir}/{split}.json'