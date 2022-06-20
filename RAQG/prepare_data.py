import jsonlines
from nltk.tokenize import word_tokenize

def uni_F1_score(pred, labels):
    pred = word_tokenize(pred)
    pred_len = len(pred)
    f1_score = 0
    for label in labels:
        label = word_tokenize(label)
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

def mix_data(raw_data_file, predictions_file):
    with jsonlines.open(raw_data_file, 'r') as reader:
        raw_data = [line for line in reader]
    with open(predictions_file, 'r') as f:
        predictions = [line.strip() for line in f.readlines()]
    assert len(raw_data) * 10 == len(predictions)
    new_data = []
    for i in range(len(raw_data)):
        example = raw_data[i]
        dialog = example['dialogue']
        gold_refs = example['query']
        preds = predictions[10 * i: 10 * (i + 1)]
        scores = [uni_F1_score(pred, gold_refs) for pred in preds]
        new_example = {'dialogue': dialog, 'query': gold_refs, 'candidate': preds, 'score': scores}
        new_data.append(new_example)
    return new_data

if __name__ == '__main__':
    data = []
    for i in [0, 1, 2]:
        data += mix_data(f'../saved_data/data_en/train_{i}_.json',
                         f'../saved_data/t5-v1_1-base-{i}f/bm_{i}_generated_predictions.txt')
    with jsonlines.open('train_en.json', 'w') as writer:
        for line in data:
            writer.write(line)