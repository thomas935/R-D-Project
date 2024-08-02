import numpy as np
import torch
from pathlib import Path

from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from config_loader import config
from Methods.Interface.Interface import LoadData, get_model_probabilities


def load_voting_data(application):
    if application == 'transformer':
        path_to_predictions = Path(f"{config['path_to_content_root']}{config['path_to_voting_data']}/Transformer/Predictions")
        path_to_labels = Path(f"{config['path_to_content_root']}{config['path_to_voting_data']}/Transformer/Labels")

        if not path_to_predictions.exists() or not path_to_labels.exists():
            raise FileNotFoundError('The path to the voting data does not exist.')
        else:
            predictions = get_model_probabilities(path_to_predictions)
            labels = np.load(path_to_labels/'labels.npy')

            return predictions, labels


def sum_vote(predictions, labels):
    sum_list = []
    for i, pair in enumerate(zip(*predictions)):
        sum_pair = np.zeros(2, dtype=float)
        for array in pair:
            sum_pair += array
        sum_list.append(np.argmax(sum_pair))
    print(sum_list)
    print(labels)
    f1 = f1_score(labels, sum_list, average='weighted')
    print(f'F1 score for sum vote: {f1}')

def majority_vote(predictions, labels):
    majority_list = []
    for i, pair in enumerate(zip(*predictions)):
        majority_pair = []
        for array in pair:
            majority_pair.append(np.argmax(array))
        majority_list.append(max(set(majority_pair), key=majority_pair.count))
    print(majority_list)
    print(labels)
    f1 = f1_score(labels, majority_list, average='weighted')
    print(f'F1 score for majority vote: {f1}')


def vote(predictions, labels):
    sum_vote(predictions, labels)
    majority_vote(predictions, labels)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    predictions, labels = load_voting_data('transformer')
    vote(predictions, labels)

if __name__ == '__main__':
    main()
