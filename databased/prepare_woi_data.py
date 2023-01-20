import re
import jsonlines
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('t5-base')
tokenizer.add_tokens(['[apprentice_persona]', '[dialog_history]', '[apprentice]', '[wizard]'])
max_turn_num = 6

def get_model_input(data):
    data = list(data.values())[0]
    persona = f"[apprentice_persona] {data['apprentice_persona'].strip()}"
    dialog = []
    model_input = []
    for idx, turn in enumerate(data['dialog_history']):
        action = turn['action'].strip()
        text = re.sub('\s+', ' ', turn['text'].strip().lower())
        if action == 'Apprentice => Wizard':
            role = 'Apprentice'
            dialog.append(f"[{role}] {text}")
        elif action == 'Wizard => Apprentice':
            role = 'Wizard'
            dialog.append(f"[{role}] {text}")
        elif action == 'Wizard => SearchAgent':
            if model_input and model_input[-1]['turn'] == len(dialog):
                if text not in model_input[-1]['query']:
                    model_input[-1]['query'].append(text)
            else:
                model_input.append({
                    'dialogue': f"{persona} [dialog_history] {' '.join(dialog[max(0, len(dialog) - max_turn_num - 1):])}".lower(),
                    'query': [text], 'turn': len(dialog)
                })
        elif action == 'SearchAgent => Wizard':
            continue
        else:
            raise Exception('UNKNOWN ACTION')
    for item in model_input:
        item.pop('turn')
    return model_input

import spacy
nlp = spacy.load('en_core_web_md')
import string
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
stop_words.update(set(string.punctuation))

print(sorted(stop_words))
def return_lemma(text):
    doc = nlp(text)
    return {token.lemma_ for token in doc if token.text not in stop_words}

def measure_over_diversity_degree(dialog, query):
    dialog_lemma = return_lemma(dialog)
    query_lemma = return_lemma(query)
    if len(query_lemma) == 0:
        print('Empty query:', query)
        d = 1
    else:
        d = len(query_lemma & dialog_lemma) / len(query_lemma)
    return d

for split in ['train', 'valid', 'test']:
    data = []
    input_len = []
    with jsonlines.open(f'../saved_data/wizard_of_interent/{split}.jsonl', 'r') as reader:
        for line in reader:
            model_inputs = get_model_input(line)
            data += model_inputs
            input_len.append(len(tokenizer.tokenize(data[-1]['dialogue'])))
    print(len(data), np.mean(input_len), max(input_len))

    for item in tqdm(data):
        item['degree'] = [measure_over_diversity_degree(item['dialogue'], q) for q in item['query']]

    output_file = f'../saved_data/data_woi/{split}.json'
    with jsonlines.open(output_file, 'w') as writer:
        for x in data:
            writer.write(x)