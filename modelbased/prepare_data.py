import os
os.environ['CUDA_VISIBLE_DEVICES'] = '6'

import jsonlines
from nltk.tokenize import word_tokenize
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm
import torch

beam = 10
lang = 'en'
degree_threshold = -1
data_file = f'../saved_data/data_woi/train.json'
model_file = f'../saved_data/t5-v1_1-base-woi'
tokenizer = AutoTokenizer.from_pretrained(model_file)
model = AutoModelForSeq2SeqLM.from_pretrained(model_file).cuda()

def uni_F1_score(pred, labels):
    def return_dict(phrase):
        word_dict = {}
        for word in phrase:
            if word not in word_dict:
                word_dict[word] = 1
            else:
                word_dict[word] += 1
        return word_dict
    pred = word_tokenize(pred)
    pred_len = len(pred)
    pred_dict = return_dict(pred)
    f1_score = 0
    for label in labels:
        label = word_tokenize(label)
        label_len = len(label)
        ref_dict = return_dict(label)
        cnt = 0

        # penalize repetition
        for word in ref_dict:
            if word in pred_dict:
                num1 = pred_dict[word]
                num2 = ref_dict[word]
                if num1 <= num2:
                    cnt += num1

        try:
            p, r = cnt / pred_len, cnt / label_len
        except:
            p, r = 0, 0
        if p == 0 or r == 0:
            f1_score = max(f1_score, 0)
        else:
            f1_score = max(f1_score, 2 * p * r/ ( p + r ))
    return f1_score

with jsonlines.open(data_file, 'r') as reader:
    raw_data = [line for line in reader]

def return_degree_token(degree):
    if lang == 'en':
        if degree_threshold <= degree:
            return '[A]'
        else:
            return '[B]'
    elif lang == 'zh':
        if degree_threshold <= degree:
            return '[A]'
        else:
            return '[B]'
    else:
        raise Exception

new_data = []
for example in tqdm(raw_data):
    dialog = example['dialogue']
    gold_refs = example['query']


    if degree_threshold > 0:
        degree_token = return_degree_token(max(example['degree']))
        model_input = tokenizer([degree_token + dialog], return_tensors='pt')
    else:
        model_input = tokenizer([dialog], return_tensors='pt')

    model_input = {k: v.cuda() for k, v in model_input.items()}
    with torch.no_grad():
        beam_search_outputs = model.generate(**model_input, num_beams=beam, output_scores=True,
                                          num_return_sequences=beam, return_dict_in_generate=True)
        sequence_outputs, sequence_scores = beam_search_outputs['sequences'], beam_search_outputs['sequences_scores']
        predictions = tokenizer.batch_decode(
            sequence_outputs, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        sorted_indices = torch.argsort(sequence_scores, descending=True)
        sequence_scores = sequence_scores.tolist()
        sorted_scores = [sequence_scores[ind] for ind in sorted_indices]
        sorted_predictions = [predictions[ind].strip() for ind in sorted_indices]

        scores = [uni_F1_score(pred, gold_refs) for pred in sorted_predictions]
        new_example = {'dialogue': dialog, 'query': gold_refs, 'candidate': sorted_predictions,
                       'score': scores, 'beam_score': sorted_scores, 'degree': example['degree']}
        new_data.append(new_example)

with jsonlines.open(f'../saved_data/data_woi/train_beam_{beam}.json', 'w') as writer:
    for line in new_data:
        writer.write(line)