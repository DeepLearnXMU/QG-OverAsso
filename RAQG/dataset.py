# coding=utf-8
import random

import jsonlines
from torch.utils.data import Dataset
import numpy as np

class QueryGenDataset(Dataset):
    def __init__(self, file_path):
        with jsonlines.open(file_path, 'r') as reader:
            self.data = [line for line in reader]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        example = self.data[item]
        return example['dialogue'], random.sample(example['query'], 1)[0]

class KDDataset(Dataset):
    def __init__(self, file_path):
        with jsonlines.open(file_path, 'r') as reader:
            self.data = [line for line in reader]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        example = self.data[item]
        return example['dialogue'], random.sample(example['candidate'], 1)[0]

class RAQGDataset(Dataset):
    def __init__(self, file_path, topk=-1, none_adjust=1.0):
        self.topk, self.none_adjust = topk, none_adjust
        print(f'Prepare data from {file_path}')
        with jsonlines.open(file_path, 'r') as reader:
            self.data = [line for line in reader]
        print(f'Before filter : {len(self.data)}')
        self.data = [x for x in self.data if not self.filter(x)]
        print(f'After filter : {len(self.data)}')
        if topk > 0:
            self.data = [self.rank_candidate(x) for x in self.data]
        if none_adjust < 1:
            self.data = [self.no_search_adjust(x) for x in self.data]
        # self.all_mean = self.calculate_mean_score()

    def calculate_mean_score(self):
        score = []
        for x in self.data:
            score += x['score']
        return np.mean(score)

    def rank_candidate(self, example):
        sorted_indices = np.argsort(-np.array(example['score'])).tolist()
        new_candidates, new_scores = [], []
        for i in sorted_indices[:self.topk]:
            new_candidates.append(example['candidate'][i])
            new_scores.append(example['score'][i])
        example['candidate'] = new_candidates
        example['score'] = new_scores
        return example

    def no_search_adjust(self, example):
        new_candidates, new_scores = [], []
        for i in range(len(example['candidate'])):
            new_candidates.append(example['candidate'][i])
            if example['candidate'][i] == '不检索':
                new_scores.append(example['score'][i] * self.none_adjust)
            else:
                new_scores.append(example['score'][i])
        example['candidate'] = new_candidates
        example['score'] = new_scores
        return example

    def filter(self, example):
        if all([s==example['score'][0] for s in example['score']]):
            return True
        return False

    def normalize(self, scores):
        scores = np.array(scores)
        mean_score = np.mean(scores)
        scores = scores - mean_score
        return scores.tolist()

    # def normalize(self, scores):
    #     return scores

    # def normalize(self, scores):
    #     scores = np.array(scores)
    #     scores = scores - self.all_mean
    #     return scores.tolist()

    # def normalize(self, scores):
    #     scores = np.array(scores)
    #     max_score = np.max(scores)
    #     min_score = np.min(scores)
    #     scores = (scores - min_score) / (max_score - min_score) - 0.5
    #     return scores.tolist()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        example = self.data[item]
        selected_index = range(len(example['candidate']))
        dialog = [example['dialogue']] * len(selected_index)
        query = [random.sample(example['query'], 1)[0]] * len(selected_index)
        candidate = [example['candidate'][i] for i in selected_index]
        score = self.normalize([example['score'][i] for i in selected_index])
        return [dialog, query, candidate, score]