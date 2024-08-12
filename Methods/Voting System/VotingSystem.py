import numpy as np
import torch
from pathlib import Path

from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from config_loader import config
from Methods.Interface.Interface import LoadData, get_model_probabilities


def load_voting_data(application):
    path_to_data = Path(f"{config['path_to_content_root']}{config['path_to_voting_data']}")

    if application == 'transformer':
        path_to_predictions = path_to_data/'Transformer/Predictions'
        path_to_labels = path_to_data/'Transformer/Labels'
        if not path_to_predictions.exists() or not path_to_labels.exists():
            raise FileNotFoundError('The path to the voting data does not exist.')
        else:
            predictions = get_model_probabilities(path_to_predictions)
            labels = np.load(path_to_labels/'labels.npy')

            return predictions, labels

    elif application == 'lstm':
        path_to_predictions = path_to_data/'LSTM/Predictions'
        path_to_labels = path_to_data/'LSTM/Labels'

        if not path_to_predictions.exists() or not path_to_labels.exists():
            raise FileNotFoundError('The path to the voting data does not exist.')
        else:
            predictions = get_model_probabilities(path_to_predictions)
            labels = np.load(path_to_labels/'labels.npy')

            return predictions, labels


def mean_vote(predictions, labels):
    mean_list = []
    for i, pair in enumerate(zip(*predictions)):
        mean_list.append(np.argmax(np.mean(pair, axis=0)))
    print(mean_list)
    f1 = f1_score(labels, mean_list, average='weighted')
    print(f'F1 score for mean vote: {f1}')

def majority_vote(predictions, labels):
    majority_list = []
    for i, pair in enumerate(zip(*predictions)):
        majority_pair = []
        for array in pair:
            majority_pair.append(np.argmax(array))
        majority_list.append(max(set(majority_pair), key=majority_pair.count))
    f1 = f1_score(labels, majority_list, average='weighted')
    print(f'F1 score for majority vote: {f1}')

def sum_square_vote(predictions, labels):
    sum_list = []
    for i, pair in enumerate(zip(*predictions)):
        sum_pair = np.zeros(2, dtype=float)
        for array in pair:
            sum_pair += array**2
        sum_list.append(np.argmax(sum_pair))

    f1 = f1_score(labels, sum_list, average='weighted')
    print(f'F1 score for sum square vote: {f1}')


def vote(predictions, labels):
    mean_vote(predictions, labels)
    majority_vote(predictions, labels)
    sum_square_vote(predictions, labels)


def model_f1_score(labels, application):
    path_to_data = Path(f"{config['path_to_content_root']}{config['path_to_voting_data']}")

    if application == 'transformer':
        path_to_directory = path_to_data/'Transformer/Predictions'
        for file in path_to_directory.iterdir():
            predictions = np.load(file)
            f1 = f1_score(labels, np.argmax(predictions, axis=1), average='weighted')
            print(f'F1 score for {file.stem}: {f1}')

    elif application == 'lstm':
        path_to_directory = path_to_data/'LSTM/Predictions'
        for file in path_to_directory.iterdir():
            predictions = np.load(file)
            f1 = f1_score(labels, np.argmax(predictions, axis=1), average='weighted')
            print(f'F1 score for {file.stem}: {f1}')

def check_labels_predictions(predictions, labels):

    print(f"Number of predictions: {len(predictions[0])}")
    print(f"Number of labels: {len(labels)}")
    print(f"Predictions: {predictions}")
    print(f"Labels: {labels}")
    if len(predictions[0]) != len(labels):
        raise ValueError('The number of predictions and labels do not match.')

def main():
    predictions, labels = load_voting_data('lstm')
    model_f1_score(labels, 'lstm')
    vote(predictions, labels)


if __name__ == '__main__':
    main()
