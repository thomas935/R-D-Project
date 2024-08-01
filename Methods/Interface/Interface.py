import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizerBase, PreTrainedTokenizerFast, PreTrainedTokenizer, AutoTokenizer, AutoModel

from config_loader import config


class CustomTransformerModel(nn.Module):
    def __init__(self, base_model, generate_embeddings):
        super(CustomTransformerModel, self).__init__()
        self.base_model = base_model
        self.generate_embeddings = generate_embeddings

        self.pre_classifier = torch.nn.Linear(base_model.config.hidden_size, config['model_class'][
            'pre-classifier_outputs'])  # Adjusted input size to match transformer output size
        self.dropout = torch.nn.Dropout(config['model_class']['dropout'])
        self.classifier = nn.Linear(config['model_class']['classifier_inputs'],
                                    config['model_class']['classifier_outputs'])

    def forward(self, input_ids, attention_mask):
        if self.generate_embeddings:
            outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
            return outputs
        else:
            outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
            cls_embedding = outputs.last_hidden_state[:, 0, :]  # [CLS] token's embedding

            # Apply additional layers
            x = self.pre_classifier(cls_embedding)
            x = F.relu(x)
            x = self.dropout(x)
            outputs = self.classifier(x)

            return outputs


class CustomDataset(Dataset):
    def __init__(self, application, **kwargs):
        """
        Initializes the dataset based on the application type.

        Parameters:
        - application (str): The type of application (e.g., 'train transformer', 'train lstm', 'train metamodel').
        - **kwargs: Additional keyword arguments for dataset initialization.
        """
        self.application = application

        if self.application == 'transformer':

            self._validate_and_assign(
                kwargs,
                {
                    'tokenizer': ((PreTrainedTokenizerBase, PreTrainedTokenizerFast, PreTrainedTokenizer), None),
                    'texts': (list, []),
                    'labels': (list, []),
                }
            )

        elif self.application == 'lstm':
            self._validate_and_assign(
                kwargs,
                {
                    'embeddings': (torch.Tensor, None),
                    'labels': (torch.Tensor, None),
                }
            )

        elif self.application == 'metamodel':
            self._validate_and_assign(
                kwargs,
                {
                    'predictions': (torch.Tensor, None),
                    'targets': (torch.Tensor, None),
                }
            )
        else:
            raise ValueError(f"Unknown application type '{application}'")

    def _validate_and_assign(self, kwargs, param_definitions):
        """
        Validate and assign parameters based on their definitions.

        Parameters:
        - kwargs (dict): Dictionary of parameters.
        - param_definitions (dict): Dictionary defining parameter names, expected types, and default values.
        """
        for param, (expected_type, default_value) in param_definitions.items():
            value = kwargs.get(param, default_value)

            # Check if the parameter is required and missing
            if value is None and default_value is None:
                raise ValueError(f"Missing required parameter: '{param}'")

            # Type check and conversion if necessary
            if not isinstance(value, expected_type) and value is not None:
                # Handling case when expected_type is a tuple
                print(f"Checking {param} with value {value}")
                if isinstance(expected_type, tuple):
                    if not any(isinstance(value, t) for t in expected_type):
                        expected_type_names = [t.__name__ for t in expected_type]
                        raise TypeError(
                            f"Parameter '{param}' must be one of the following types: {', '.join(expected_type_names)}, but got {type(value).__name__}."
                        )
                else:
                    if expected_type == torch.Tensor and not isinstance(value, torch.Tensor):
                        try:
                            value = torch.tensor(value, dtype=torch.float32)
                        except Exception as e:
                            raise TypeError(
                                f"Parameter '{param}' must be convertible to torch.Tensor, but encountered an error: {str(e)}."
                            )
                    else:
                        raise TypeError(
                            f"Parameter '{param}' must be of type {expected_type.__name__}, but got {type(value).__name__}."
                        )

            setattr(self, param, value)

    def __getitem__(self, index):
        """
        Return the dataset item based on the application type.

        Parameters:
        - index (int): The index of the dataset item to retrieve.
        """
        if self.application == 'transformer':
            text = self.texts[index]
            label = torch.tensor(self.labels[index])

            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding="max_length",
                                    max_length=config['max_length'])
            ids = inputs['input_ids'].squeeze(0)
            mask = inputs['attention_mask'].squeeze(0)

            return {
                'ids': torch.tensor(ids),
                'mask': torch.tensor(mask),
                'labels': torch.tensor(label)
            }

        elif self.application == 'lstm':
            return {
                'embeddings': self.embeddings[index],
                'labels': self.labels[index]
            }
        elif self.application == 'metamodel':
            return {
                'predictions': self.predictions[index],
                'targets': self.targets[index]
            }
        else:
            raise ValueError(f"Unknown application type '{self.application}'")

    def __len__(self):
        """
        Return the length of the dataset based on the application type.
        """
        if self.application == 'transformer':
            return len(self.texts)
        elif self.application == 'lstm':
            return len(self.embeddings)
        elif self.application == 'metamodel':
            return len(self.predictions)
        else:
            raise ValueError(f"Unknown application type '{self.application}'")

    def __str__(self):
        # print len of attribut if it exists
        if hasattr(self, 'texts'):
            print(f"Length of texts: {len(self.texts)}")
        if hasattr(self, 'labels'):
            print(f"Length of labels: {len(self.labels)}")
        if hasattr(self, 'embeddings'):
            print(f"Length of embeddings: {len(self.embeddings)}")
        if hasattr(self, 'predictions'):
            print(f"Length of predictions: {len(self.predictions)}")
        if hasattr(self, 'targets'):
            print(f"Length of targets: {len(self.targets)}")
        return ""


class LoadData:
    def __init__(self, application, model_name, percentage_of_data=None):
        self.application = application
        self.model_name = model_name
        self.percentage_of_data = percentage_of_data

        if self.application == 'transformer':

            self.load_transformer_data()
            self.tokenizer = load_tokenizer(config['transformer']['tokenizer'][f'{self.model_name}'])

            train_params = {
                'tokenizer': self.tokenizer,
                'texts': self.train_texts,
                'labels': self.train_labels,
            }
            self.train_set = CustomDataset(application=application, **train_params)

            test_params = {
                'tokenizer': self.tokenizer,
                'texts': self.test_texts,
                'labels': self.test_labels,
            }
            self.test_set = CustomDataset(application=application, **test_params)

        elif self.application == 'lstm':

            self.load_lstm_data(model_name)
            train_params = {
                'embeddings': self.embedding_train,
                'labels': self.labels_train,
            }
            self.train_set = CustomDataset(application=application, **train_params)

            test_params = {
                'embeddings': self.embedding_test,
                'labels': self.labels_test,
            }
            self.test_set = CustomDataset(application=application, **test_params)

        elif self.application == 'metamodel':

            self.load_metamodel_data()
            train_params = {
                'predictions': self.train_predictions,
                'targets': self.train_targets,
            }
            self.train_set = CustomDataset(application=application, **train_params)

            test_params = {
                'predictions': self.test_predictions,
                'targets': self.test_targets,
            }
            self.test_set = CustomDataset(application=application, **test_params)

    def load_transformer_data(self):
        def adapt_data(df):
            df['list'] = df[df.columns[1]].values.tolist()
            df['list'] = [[1 if i == x else 0 for i in range(2)] for x in
                          df['list']]  # Modify label to binary encoding

            new_df = df[['text', 'list']].copy()
            new_df.columns = ['text', 'sexist_label']
            if self.percentage_of_data:
                new_df = new_df.sample(frac=self.percentage_of_data, random_state=42)
                new_df = new_df.reset_index(drop=True)
            return new_df

        path_to_train = Path(f"{config['path_to_content_root']}{config['transformer']['path_to_train']}")
        name_train_file = 'train_dataset.csv'
        path_to_test = Path(f"{config['path_to_content_root']}{config['transformer']['path_to_test']}")
        name_test_file = 'test_dataset.csv'

        if os.path.exists(path_to_train):
            print(f'Loading data from {path_to_train}')
            train_df = pd.read_csv(path_to_train / name_train_file)
            train_df = adapt_data(train_df)
        else:
            raise ValueError(f'Path to train dataset {path_to_train} does not exist.')

        if os.path.exists(path_to_test):
            print(f'Loading data from {path_to_test}')
            test_df = pd.read_csv(path_to_test / name_test_file)
            test_df = adapt_data(test_df)
        else:
            raise ValueError(f"Path to test dataset '{path_to_test}' does not exist.")

        self.train_texts = train_df['text'].tolist()
        self.train_labels = train_df['sexist_label'].tolist()
        self.test_texts = test_df['text'].tolist()
        self.test_labels = test_df['sexist_label'].tolist()

        print(f"Train set: {len(self.train_texts)} samples")
        print(f"Test set: {len(self.test_texts)} samples")

    def load_lstm_data(self, model_name):
        embedding_path = Path(config['LSTM']['embeddings']['path_to_embedding'])
        label_path = Path(config['LSTM']['path_to_label'])

        if os.path.exists(label_path):
            self.labels_train = torch.load(np.load(label_path / 'Train' / f'{model_name}_train_labels.npy').flatten())
            self.labels_train = torch.nn.functional.one_hot(
                self.labels_train,
                num_classes=config['model_class']['classifier_outputs']
            ).to(torch.float32)
            embedding_train_shape = config['LSTM']['embeddings']['embeddings_shape']
            embedding_train_shape[0] = len(self.labels_train)

            self.labels_test = torch.load(np.load(label_path / 'Test' / f'{model_name}_test_labels.npy').flatten())
            self.labels_test = torch.nn.functional.one_hot(
                self.labels_test,
                num_classes=config['model_class']['classifier_outputs']
            ).to(torch.float32)
            embedding_test_shape = len(self.labels_test)
        else:
            raise ValueError(f"Path to labels '{label_path}' does not exist.")

        if os.path.exists(embedding_path):
            self.embedding_train = torch.tensor(
                np.load(embedding_path / 'Train' / f'{model_name}_train_embeddings.npy').reshape(embedding_train_shape),
                dtype=torch.float32)
            self.embedding_test = torch.tensor(
                np.load(embedding_path / 'Test' / f'{model_name}_test_embeddings.npy').reshape(embedding_test_shape),
                dtype=torch.float32)

        else:
            raise ValueError(f"Path to embeddings '{embedding_path}' does not exist.")

    def load_metamodel_data(self):
        def concatenate_probabilities(directory):
            arr = []
            for file_path in sorted(directory.iterdir()):
                temp_arr = np.load(file_path)
                arr.append(temp_arr)
            arr = np.hstack(arr)

            return arr

        path_to_predictions = Path(config['metamodel']['path_to_predictions'])
        path_to_targets = Path(config['metamodel']['path_to_targets'])

        if os.path.exists(path_to_predictions):
            probabilities = concatenate_probabilities(path_to_predictions)
        else:
            raise ValueError(f"Path to predictions '{path_to_predictions}' does not exist.")

        if os.path.exists(path_to_targets):
            targets = np.load(path_to_targets / 'targets.npy')
        # randomly select 80% of the data for training and 20% for testing
        train_size = int(0.8 * len(probabilities))
        self.train_predictions = torch.tensor(probabilities[:train_size], dtype=torch.float32)
        self.train_targets = torch.tensor(targets[:train_size], dtype=torch.float32)
        self.test_predictions = torch.tensor(probabilities[train_size:], dtype=torch.float32)
        self.test_targets = torch.tensor(targets[train_size:], dtype=torch.float32)

    def __getattr__(self):
        return self.train_set, self.test_set

    def __getitem__(self, index):
        params = {
            'texts': self.train_texts[index],
            'labels': self.train_labels[index]
        }
        return params


def load_tokenizer(tokenizer_name):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    return tokenizer


def check_dataloader(dataloader: DataLoader):
    num_batches_to_print = 1

    # Iterate over the DataLoader and print batches
    for batch_idx, batch in enumerate(dataloader):
        print(f"Batch {batch_idx + 1}:")
        print(batch)
        # Break after printing the desired number of batches
        if batch_idx + 1 == num_batches_to_print:
            break


def check_dataset(dataset: CustomDataset):
    print(dataset.application)
    for i in range(len(dataset)):
        print(dataset[i])
        if i == 5:
            break


def initialise_model(model_name: str, device: torch.device, embedding=False):
    base_model = AutoModel.from_pretrained(config['transformer']['tokenizer'][f'{model_name}'],
                                           output_hidden_states=True)
    model = CustomTransformerModel(base_model, embedding)
    return model


def load_model(model_name: str, device: torch.device, embedding=False):
    model = initialise_model(model_name, device, embedding)
    model_path = f"{config['path_to_content_root']}{config['transformer']['path_to_model_trained']}/{model_name}.pth"
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    return model
