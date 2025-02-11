import os
from collections import Counter
from pathlib import Path
from typing import Dict, Any, Tuple

import lightning as L
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from lightning.pytorch.loggers import WandbLogger
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader
from torcheval.metrics import BinaryF1Score
from transformers import PreTrainedTokenizerBase, PreTrainedTokenizerFast, PreTrainedTokenizer, AutoTokenizer, AutoModel

from config_loader import config


class MetaModel(L.LightningModule):
    def __init__(self, hidden_size, num_layers):
        super(MetaModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        layers = []
        # Input layer
        layers.append(nn.Linear(6, hidden_size))
        layers.append(nn.ReLU())

        # Hidden layers
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())

        # Output layer
        layers.append(nn.Linear(hidden_size, 1))
        layers.append(nn.Sigmoid())

        self.model = nn.Sequential(*layers)

        self.criterion = nn.BCELoss()
        self.metric = BinaryF1Score()

        self.predictions = torch.tensor([])
        self.labels = torch.tensor([])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = self.model(x)
        return outputs.squeeze(1)

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        x, y = batch
        y_hat = self(x)

        loss = self.criterion(y_hat, y)
        self.log("val_loss", loss)

        self.metric.update(y_hat, y)
        self.log("Val_F1_score", self.metric.compute())

    def on_validation_epoch_end(self) -> None:
        self.metric.reset()

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = torch.optim.Adam(self.parameters(), lr=float(config['model_parameters']['learning_rate']))

        scheduler = ReduceLROnPlateau(optimizer,
                                      mode=config['scheduler']['mode'],
                                      factor=float(config['scheduler']['factor']),
                                      patience=int(config['scheduler']['patience']),
                                      threshold=float(config['scheduler']['threshold']))
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': config['scheduler']['interval'],
                'frequency': config['scheduler']['frequency'],
                'monitor': config['scheduler']['monitor'],
                'strict': config['scheduler']['strict']
            }
        }

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        x, y = batch
        pred = self(x)
        loss = self.criterion(pred, y)
        self.log("test_loss", loss)

        # Concatenate predictions and labels
        self.predictions = torch.cat([self.predictions, pred])
        self.labels = torch.cat([self.labels, y])

    def on_test_end(self) -> None:
        self.calculate_f1_score()
        save_model(self, 'metamodel', 'metamodel')

    def calculate_f1_score(self):
        # Pass predictions and labels to tensor

        self.metric.update(self.predictions, self.labels)
        f1 = self.metric.compute()
        print(f"F1 Score: {f1}")
        self.logger.experiment.log({"F1 Score": f1})


class CustomLSTMModel(L.LightningModule):
    def __init__(self, hidden_dim: int, num_layers: int, model_name: str):
        super(CustomLSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.model_name = model_name

        self.lstm = nn.LSTM(768, self.hidden_dim, self.num_layers, batch_first=True)
        self.fc = nn.Linear(self.hidden_dim, 2)

        self.criterion = nn.BCEWithLogitsLoss()
        self.metric = BinaryF1Score()

        self.predictions = torch.tensor([])
        self.labels = torch.tensor([])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)
        lstm_out = self.fc(lstm_out[:, -1, :])
        return lstm_out

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        Training step.

        Parameters:
        - batch (Tuple[torch.Tensor, torch.Tensor]): Input and target tensors.
        - batch_idx (int): Index of the batch.

        Returns:
        - torch.Tensor: Loss value.
        """
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """
        Validation step.

        Parameters:
        - batch (Tuple[torch.Tensor, torch.Tensor]): Input and target tensors.
        - batch_idx (int): Index of the batch.
        """
        x, y = batch
        y_hat = self(x)

        loss = self.criterion(y_hat, y)
        self.log("val_loss", loss)

        y_hat = torch.argmax(y_hat, dim=1)
        y = torch.argmax(y, dim=1)

        self.metric.update(y_hat, y)
        self.log("Val_F1_score", self.metric.compute())

    def on_validation_epoch_end(self) -> None:
        """
        Called at the end of validation epoch.
        """
        self.metric.reset()

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """
        Test step.

        Parameters:
        - batch (Tuple[torch.Tensor, torch.Tensor]): Input and target tensors.
        - batch_idx (int): Index of the batch.
        """
        x, y = batch
        pred = self(x)
        loss = self.criterion(pred, y)
        self.log("test_loss", loss)

        # Concatenate predictions and labels
        self.predictions = torch.cat([self.predictions, pred])
        self.labels = torch.cat([self.labels, y])

    def configure_optimizers(self) -> Dict[str, Any]:
        """
        Configures optimizers and learning rate schedulers.

        Returns:
        - Dict[str, Any]: Optimizer and scheduler configuration.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=float(config['model_parameters']['learning_rate']))

        scheduler = ReduceLROnPlateau(optimizer,
                                      mode=config['scheduler']['mode'],
                                      factor=float(config['scheduler']['factor']),
                                      patience=int(config['scheduler']['patience']),
                                      threshold=float(config['scheduler']['threshold']))
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': config['scheduler']['interval'],
                'frequency': config['scheduler']['frequency'],
                'monitor': config['scheduler']['monitor'],
                'strict': config['scheduler']['strict']
            }
        }

    def on_test_end(self) -> None:
        self.calculate_f1_score()

        save_model(self, f'metamodel_{self.hidden_dim}_{self.num_layers}', 'lstm')

    def calculate_f1_score(self):
        # Pass predictions and labels to tensor
        self.predictions = torch.argmax(self.predictions, dim=1)
        self.labels = torch.argmax(self.labels, dim=1)

        self.metric.update(self.predictions, self.labels)
        f1 = self.metric.compute()
        print(f"F1 Score: {f1}")
        self.logger.experiment.log({"F1 Score": f1})

    def save(self):
        predictions = self.predictions.cpu().detach().numpy()
        labels = self.labels.cpu().detach().numpy()

        predictions = np.argmax(predictions, axis=1)
        labels = np.argmax(labels, axis=1)

        path_to_save = Path(config['path_to_content_root']) / config['path_to_save']
        path_to_save.mkdir(parents=True, exist_ok=True)
        # mkdir for predictions and labels
        (path_to_save / 'Predictions').mkdir(parents=True, exist_ok=True)
        (path_to_save / 'Labels').mkdir(parents=True, exist_ok=True)

        np.save(
            path_to_save / 'Predictions' / f'Model_{self.model_name}_{self.num_layers}_{self.hidden_dim}_predictions.npy',
            predictions)
        np.save(path_to_save / 'Labels' / f'labels.npy', labels)


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
                    'labels': (torch.Tensor, None),
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
            return self.embeddings[index], self.labels[index]

        elif self.application == 'metamodel':
            return self.predictions[index], self.labels[index]

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
            print(f"Labels: {self.labels}")
        if hasattr(self, 'embeddings'):
            print(f"Length of embeddings: {len(self.embeddings)}")
            print(f"Embeddings: {self.embeddings}")
        if hasattr(self, 'predictions'):
            print(f"Length of predictions: {len(self.predictions)}")
        if hasattr(self, 'labels'):
            print(f"Length of labels: {len(self.labels)}")
        return ""


class LoadData:
    def __init__(self, application, model_name=None, percentage_of_data=None, on_approach=None):
        self.application = application
        self.model_name = model_name
        self.percentage_of_data = percentage_of_data
        self.on_approach = on_approach

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
                'embeddings': self.embeddings_train,
                'labels': self.labels_train,
            }
            self.train_set = CustomDataset(application=application, **train_params)

            train_size = int(config['model_parameters']['train_size'] * len(self.train_set))
            val_size = len(self.train_set) - train_size

            self.train_set, self.val_set = torch.utils.data.random_split(self.train_set, [train_size, val_size])

            test_params = {
                'embeddings': self.embeddings_test,
                'labels': self.labels_test,
            }
            self.test_set = CustomDataset(application=application, **test_params)

        elif self.application == 'metamodel':
            self.load_metamodel_data()

            train_params = {
                'predictions': self.train_predictions,
                'labels': self.train_labels,
            }
            self.train_set = CustomDataset(application=self.application, **train_params)

            val_params = {
                'predictions': self.val_predictions,
                'labels': self.val_labels,
            }
            self.val_set = CustomDataset(application=self.application, **val_params)

            test_params = {
                'predictions': self.test_predictions,
                'labels': self.test_labels,
            }
            self.test_set = CustomDataset(application=self.application, **test_params)

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
        embedding_path = f"{config['path_to_content_root']}{config['path_to_embeddings']}"
        label_path = f"{config['path_to_content_root']}{config['path_to_labels']}"

        if os.path.exists(label_path):
            self.labels_train = np.load(f"{label_path}/Train/{model_name}_labels.npy")
            self.labels_train = torch.tensor(self.labels_train, dtype=torch.float32)

            embeddings_train_shape = config['embedding_shape'].copy()
            embeddings_train_shape.insert(0, len(self.labels_train))
            print(embeddings_train_shape)

            self.labels_test = np.load(f"{label_path}/Test/{model_name}_labels.npy")
            self.labels_test = torch.tensor(self.labels_test, dtype=torch.float32)

            embeddings_test_shape = config['embedding_shape'].copy()
            embeddings_test_shape.insert(0, len(self.labels_test))
            print(embeddings_test_shape)

        else:
            raise ValueError(f"Path to labels '{label_path}' does not exist.")

        if os.path.exists(embedding_path):
            self.embeddings_train = np.load(f"{embedding_path}/Train/{model_name}_embeddings.npy").reshape(
                embeddings_train_shape)
            self.embeddings_train = torch.tensor(self.embeddings_train, dtype=torch.float32)

            self.embeddings_test = np.load(f"{embedding_path}/Test/{model_name}_embeddings.npy").reshape(
                embeddings_test_shape)
            self.embeddings_test = torch.tensor(self.embeddings_test, dtype=torch.float32)

        else:
            raise ValueError(f"Path to embeddings '{embedding_path}' does not exist.")

    def load_metamodel_data(self):

        path_to_data = f"{config['path_to_content_root']}{config['path_to_data']}"

        if self.on_approach == 'transformer':
            path_to_predictions = f"{path_to_data}/Transformer/Predictions"
            path_to_labels = f"{path_to_data}/Transformer/Labels"
        elif self.on_approach == 'lstm':
            path_to_predictions = f"{path_to_data}/LSTM/Predictions"
            path_to_labels = f"{path_to_data}/LSTM/Labels"
        else:
            raise ValueError(f"Unknown application type '{self.application}'")

        if os.path.exists(path_to_predictions):
            probabilities = concatenate_probabilities(Path(path_to_predictions))
        else:
            raise ValueError(f'Path to predictions {path_to_predictions} does not exist.')

        if os.path.exists(path_to_labels):
            labels = np.load(f"{path_to_labels}/labels.npy")
        else:
            raise ValueError(f"Path to labels '{path_to_labels}' does not exist.")

        # randomly select 70% of the data for training, 10% for validation and 20% for testing
        train_size = int(config['train_size'] * len(probabilities))
        val_size = int(config['val_size'] * len(probabilities))

        self.train_predictions = torch.tensor(probabilities[:train_size], dtype=torch.float32)
        self.val_predictions = torch.tensor(probabilities[train_size:train_size + val_size], dtype=torch.float32)
        self.test_predictions = torch.tensor(probabilities[train_size + val_size:], dtype=torch.float32)

        self.train_labels = torch.tensor(labels[:train_size], dtype=torch.float32)
        self.val_labels = torch.tensor(labels[train_size:train_size + val_size], dtype=torch.float32)
        self.test_labels = torch.tensor(labels[train_size + val_size:], dtype=torch.float32)


    def __getattr__(self, items=None):
        if items == 'train_set':
            return self.train_set
        elif items == 'test_set':
            return self.test_set
        elif self.application == 'lstm' or self.application == 'metamodel':
            return self.train_set, self.val_set, self.test_set
        else:
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
    for i in range(5):
        print(next(iter(dataset)))

    # Extract labels from the dataset
    labels = [entry['labels'].item() for entry in dataset]

    # Count occurrences of each label
    label_counts = Counter(labels)

    # Print the occurrence of each label
    for label, count in label_counts.items():
        print(f"Label {label}: {count} occurrences")


def initialise_model(application: str, model_name=None, embedding=None, params=None):
    if application == 'transformer':
        base_model = AutoModel.from_pretrained(config['transformer']['tokenizer'][f'{model_name}'],
                                               output_hidden_states=True)
        model = CustomTransformerModel(base_model, embedding)
        return model

    elif application == 'lstm':
        model = CustomLSTMModel(params['hidden_dim'], params['num_layers'], params['model_name'])
        return model

    elif application == 'metamodel':
        model = MetaModel(params['hidden_dim'], params['num_layers'])
        return model


def load_model(model_name: str, application: str, embedding=False):
    if application == 'transformer':
        model = initialise_model(application, model_name, embedding)
        model_path = f"{config['path_to_content_root']}{config['transformer']['path_to_model_trained']}/{model_name}.pth"
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        return model


def get_outputs(batch, model, device):
    ids = batch['ids'].to(device, dtype=torch.long)
    mask = batch['mask'].to(device, dtype=torch.long)
    label = batch['labels'].to(device, dtype=torch.float)

    with torch.no_grad():
        outputs = model(ids, mask)
    return outputs, label


def concatenate_probabilities(directory):
    arr = []
    for file_path in sorted(directory.iterdir()):
        temp_arr = np.load(file_path)
        arr.append(temp_arr)
    arr = np.hstack(arr)

    return arr


def get_model_probabilities(directory):
    list_of_arrays = []
    for file_path in sorted(directory.iterdir()):
        temp_arr = np.load(file_path)
        list_of_arrays.append(temp_arr)

    return list_of_arrays


def save_model(model, model_name, application):
    if application == 'transformer':
        model_path = f"{config['path_to_content_root']}{config['transformer']['path_to_model_trained']}/{model_name}.pth"
        torch.save(model.state_dict(), model_path)
    elif application == 'lstm':
        model_path = f"{config['path_to_content_root']}{config['path_to_model_trained']}/{model_name}.pth"
        torch.save(model.state_dict(), model_path)
    else:
        model_path = f"{config['path_to_content_root']}{config['path_to_model_trained']}/metamodel.pth"

        torch.save(model.state_dict(), model_path)


def training_model(train_dataloader, val_dataloader, test_dataloader, device, application, model_name=None):
    for hidden_dim in config['model_parameters']['train_hidden_dimensions']:
        for num_layers in config['model_parameters']['train_numbers_layers']:

            params = {
                'hidden_dim': hidden_dim,
                'num_layers': num_layers,
                'model_name': model_name,
            }

            model = initialise_model(application, params=params)
            model.to(device)

            trainer = L.Trainer(
                max_epochs=config['model_parameters']['max_epochs'],
                logger=WandbLogger(),  # Logging with WandB
            )
            if model_name is None:
                name = f'{hidden_dim}_{num_layers}'
            else:
                name = f'{model_name}_{hidden_dim}_{num_layers}'
            # Initialize WandB run

            wandb.init(
                project=config['wandb']['project'],
                # check if model_name is None
                name=name,
                settings=wandb.Settings(quiet=True)
            )

            try:

                trainer.fit(model, train_dataloader, val_dataloader)

                trainer.test(model, test_dataloader)

            except Exception as e:
                raise e

            finally:
                wandb.finish()
