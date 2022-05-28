import jsonlines
from nltk.tokenize import word_tokenize as nltk_word_tokenize
from nltk.corpus import stopwords
import string

from tqdm import tqdm

stop_words = set(stopwords.words('english'))
stop_words.update(set(string.punctuation))

import spacy
spacy.require_gpu()
nlp = spacy.load('en_core_web_md')

def word_tokenize(text):
    doc = nlp(text)
    tokens = [token.text for token in doc]
    lemmas = [token.lemma_ for token in doc]
    return tokens, lemmas

unformat = 0
updated = 0

def uni_F1_score(pred, labels):
    global unformat
    if '|' in pred:
        pred = pred.split('|')[1].strip()
    elif pred != 'none':
        # raise Exception(f'strange pred {pred}')
        unformat += 1
    else:
        return 0
    pred = nltk_word_tokenize(pred)
    pred_len = len(pred)
    f1_score = 0
    for label in labels:
        if '|' in label:
            label = label.split('|')[1].strip()
        elif label != 'none':
            raise Exception(f'strange label {label}')
        label = nltk_word_tokenize(label)
        label_len = len(label)
        common_len = len(set(pred) & set(label))
        try:
            p, r = common_len / pred_len, common_len / label_len
        except:
            p, r = 0, 0
        if p == 0 or r == 0:
            f1_score = max(f1_score, 0)
        else:
            f1_score = max(f1_score, 2*p*r/(p+r))
    return f1_score

def clean_predictions(dialog, queries):
    context_tokens, context_lemmas = word_tokenize(dialog)
    mixed_queries = set()
    global updated
    for query in queries:
        if query != 'none':
            if '|' not in query:
                continue
            raw_query = query
            template = []
            query = query.split('|')[1].strip()
            if len(query) == 0:
                continue
            query_tokens, query_lemmas = word_tokenize(query)
            for qt, ql in zip(query_tokens, query_lemmas):
                if ql in context_lemmas or qt in stop_words:
                    template.append(qt)
                elif len(template) == 0 or template[-1] != '<extra_id_0>':
                    template.append('<extra_id_0>')
            updated_query = f"{' '.join(template)} | {query}"
            if updated_query != raw_query:
                updated += 1
            mixed_queries.add(updated_query)
    if len(mixed_queries) == 0:
        mixed_queries.add('none')
    return list(mixed_queries)

def mix_data(raw_data_file, predictions_file):
    with jsonlines.open(raw_data_file, 'r') as reader:
        raw_data = [line for line in reader]
    with open(predictions_file, 'r') as f:
        predictions = [line.strip() for line in f.readlines()]
    assert len(raw_data) * 10 == len(predictions)
    new_data = []
    for i in tqdm(range(len(raw_data))):
        example = raw_data[i]
        dialog = example['dialogue']
        gold_refs = example['query']
        raw_preds = predictions[10 * i: 10 * (i + 1)]
        preds = clean_predictions(dialog, raw_preds)
        scores = [uni_F1_score(pred, gold_refs) for pred in preds]
        new_example = {'dialogue': dialog, 'query': gold_refs, 'candidate': preds, 'score': scores}
        new_data.append(new_example)
    return new_data

if __name__ == '__main__':
    data = []
    for i in [0, 1, 2]:
        data += mix_data(f'../woi_data_tpl_gen_3f/train_{i}_.json',
                         f'../t5-v1_1-base-tpl-gen-{i}f/bm_{i}_generated_predictions.txt')
    print(unformat, updated)
    with jsonlines.open('train_tpl_en.json', 'w') as writer:
        for line in data:
            writer.write(line)