# coding=utf-8
import os
import re
import jsonlines
from string import Template
from transformers import AutoTokenizer
import numpy as np

tokenizer = AutoTokenizer.from_pretrained('Langboat/mengzi-t5-base')

stopwords = set()
with open('cn_stopwords.txt', 'r') as f:
    for k in f.readlines():
        stopwords.add(k.strip())

def word_tokenize(text):
    text = re.sub('\s+', '', text)
    return [x for x in tokenizer.tokenize(text) if x != tokenizer.convert_ids_to_tokens([259])[0]]

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
        if maxNum > 0 and not all([True if x in stopwords else False for x in s1[i - maxNum: i]]):
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
    cur_str = ''
    for x, y in zip(s1, labels):
        if y == 'I':
            cur_str += x
            cur_str = cur_str.strip()
        elif cur_str:
            output.append(cur_str)
            cur_str = ''
    if cur_str:
        output.append(cur_str)
    return output


# dialogue history -> search query
def preprocess(file_path):
    data = []
    template = Template('${role}：${text}')
    length, target_length = [], []
    with jsonlines.open(file_path, 'r') as reader:
        for line in reader:
            dialog = []
            background = ''
            topic = []
            for _topic in line['user_topical']:
                _topic = _topic.strip().lower()
                if _topic:
                    topic.append(_topic)
            position = line['user_location'].strip().lower()
            topic = re.sub('\s+', '', '，'.join(topic))
            position = re.sub('\s+', '', position)
            if topic:
                background += f'话题：{topic} '
            if position:
                background += f'位置：{position} '
            if background:
                dialog.append(background.strip())
            for utterance in line['conversation']:
                role, text = utterance['role'], re.sub('\s+', '', utterance['utterance']).lower()
                if role == 'bot':
                    role = '机器人'
                else:
                    role = '用户'
                if role == 'bot' or role == '机器人':
                    if 'use_query' in utterance and utterance['use_query'].strip():
                        query = utterance['use_query'].strip().lower()
                        query = re.sub('\s+', '', query)
                        # clean repeat
                        if query[:len(query) // 2] == query[len(query) // 2:] and query != '哔哩哔哩':
                            query = query[:len(query) // 2]
                        query = '检索' + query
                    else:
                        query = '不检索'
                    query_cache = {query}
                    if 'other_search' in utterance:
                        for cand_query in utterance['other_search']:
                            cand_query = cand_query['search_query'].strip().lower()
                            cand_query = re.sub('\s+', '', cand_query)
                            if cand_query:
                                # clean repeat
                                if cand_query[:len(cand_query) // 2] == cand_query[len(cand_query) // 2:] \
                                   and cand_query != '哔哩哔哩':
                                    cand_query = cand_query[:len(cand_query) // 2]
                                cand_query = '检索' + cand_query
                                if cand_query not in query_cache:
                                    query_cache.add(cand_query)
                    data.append({'dialogue': ' '.join(dialog), 'query': list(query_cache)})
                dialog.append(template.substitute({'role': role, 'text': text}))

            length.append(len(tokenizer.tokenize(' '.join(dialog))))
            target_length.append(len(tokenizer.tokenize(query)))
    print(max(length), max(target_length))
    print(len(data))
    return data


if __name__ == '__main__':
    k_fold = 5
    output_dir = f'../../saved_data/data_4mz_{k_fold}f'
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    for split in ['dev', 'train']:
        file_path = f'../../saved_data/DuSinc_release/{split}.txt'
        output_path = f'{output_dir}/{split}.json'
        data = preprocess(file_path)
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
        else:
            output_file = f'{output_dir}/{split}.json'
            with jsonlines.open(output_file, 'w') as writer:
                for x in data:
                    writer.write(x)

