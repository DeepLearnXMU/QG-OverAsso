# coding=utf-8
import os
import re
import jsonlines
from string import Template
from transformers import AutoTokenizer
import numpy as np

tokenizer = AutoTokenizer.from_pretrained('Langboat/mengzi-t5-base')
tokenizer.add_tokens(['[话题]', '[位置]', '[用户]', '[机器]', '[对话]'])
max_turn_num = 6
multi_ref = False

def clean(string):
    return re.sub('\s+', '', string.strip()).lower()

def return_degree(dialog, query):
    dialog = set(dialog)
    query = set(query)
    return len(dialog & query) / len(query)

# dialogue history -> search query
def preprocess(file_path):
    data = []
    template = Template('[${role}] ${text}')
    dialog_length, query_length = [], []
    with jsonlines.open(file_path, 'r') as reader:
        for line in reader:
            dialog = []
            raw_dialog_tokens = []
            background = f'[话题] {clean(" ".join(line["user_topical"]))} [位置] {clean(line["user_location"])}'
            raw_dialog_tokens += " ".join(line["user_topical"]).split() + line["user_location"].split()
            for utterance in line['conversation']:
                role, text = utterance['role'], clean(utterance['utterance'])
                if role == 'bot':
                    role = '机器'
                else:
                    role = '用户'
                if role == 'bot' or role == '机器':
                    query_list = []
                    raw_query_list = []
                    if 'use_query' in utterance:
                        query = clean(utterance['use_query'])
                        if query:
                            # clean repeat
                            if query[:len(query) // 2] == query[len(query) // 2:] and query != '哔哩哔哩':
                                query = query[:len(query) // 2]
                            query_list.append(query)
                            raw_query_list.append(utterance['use_query'].split())
                    if 'other_search' in utterance and multi_ref:
                        for _cand_query in utterance['other_search']:
                            cand_query = clean(_cand_query['search_query'])
                            if cand_query:
                                # clean repeat
                                if cand_query[:len(cand_query) // 2] == cand_query[len(cand_query) // 2:] \
                                   and cand_query != '哔哩哔哩':
                                    cand_query = cand_query[:len(cand_query) // 2]
                                if cand_query not in query_list:
                                    query_list.append(cand_query)
                                    raw_query_list.append(_cand_query['search_query'].split())
                    if query_list:
                        degree = [return_degree(raw_dialog_tokens, raw_q) for _, raw_q in zip(query_list, raw_query_list)]
                        data.append({'dialogue': f"{background} [对话] {' '.join(dialog[max(0, len(dialog) - max_turn_num - 1):])}",
                                     'query': query_list, 'degree': degree})
                dialog.append(template.substitute({'role': role, 'text': text}))
                raw_dialog_tokens += utterance['utterance'].split()
    for item in data:
        dialog_length.append(len(tokenizer.tokenize(item['dialogue'])))
        query_length += [len(tokenizer.tokenize(query)) for query in item['query']]
    print(len(data))
    print(max(dialog_length), max(query_length), np.mean(dialog_length), np.mean(query_length))
    return data


if __name__ == '__main__':
    k_fold = 3
    output_dir = '../saved_data/data_dusinc'
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    file_path = f'../saved_data/DuSinc_release/train.txt'
    data = preprocess(file_path)
    for i in range(k_fold):
        with jsonlines.open(f'{output_dir}/train_static_view_{i}th_fold.json', 'w') as writer:
            for j in range(len(data)):
                if j % k_fold != i:
                    writer.write(data[j])
        with jsonlines.open(f'{output_dir}/support_static_view_{i}th_fold.json', 'w') as writer:
            for j in range(len(data)):
                if j % k_fold == i:
                    writer.write(data[j])

