from typing import Dict, Any

from dotenv import load_dotenv

load_dotenv(verbose=True)

from config_loader import config

from typing import Tuple

import torch
import numpy as np
import pandas as pd
import wandb

from torch import nn
import lightning as L
from torch.utils.data import Dataset, DataLoader, random_split
from lightning.pytorch.loggers import WandbLogger
from pathlib import Path

from torch.optim.lr_scheduler import ReduceLROnPlateau
from lightning.pytorch.callbacks import Callback
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

from torcheval.metrics import BinaryF1Score, MulticlassF1Score

# Seed the random number generators
L.seed_everything(42)


class CommentDataset(Dataset):
    def __init__(self, embeddings: torch.Tensor, labels: torch.Tensor):
        """
        Initializes the CommentDataset.

        Parameters:
        - embeddings (torch.Tensor): Tensor containing the embeddings.
        - labels (torch.Tensor): Tensor containing the labels.
        """
        self.embeddings = embeddings
        self.labels = labels

    def __len__(self) -> int:
        """
        Returns the number of samples in the dataset.
        """
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns a single sample from the dataset.

        Parameters:
        - idx (int): Index of the sample to retrieve.

        Returns:
        - Tuple[torch.Tensor, torch.Tensor]: Embedding and label of the sample.
        """
        return self.embeddings[idx], self.labels[idx]

    def __str__(self) -> str:
        """
        Returns a string representation of the dataset.

        Returns:
        - str: First two embeddings and labels.
        """
        return f'Embeddings: {self.embeddings[:2]}\nLabels: {self.labels[:2]}'

    @staticmethod
    def data_load(model_name: str) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Loads data for the specified model and prepares DataLoaders for training, validation, and test sets.

        Parameters:
        - model_name (str): Name of the model to load data for.

        Returns:
        - Tuple[DataLoader, DataLoader, DataLoader]: DataLoaders for training, validation, and test datasets.
        """
        embeddings_name = 'embeddings.npy'  # Name of the embeddings file
        labels_name = 'labels.npy'  # Name of the labels file

        # Initialize DataLoaders
        train_dataloader, val_dataloader, test_dataloader = None, None, None

        for step in config['steps']:
            # Construct paths to embeddings and labels files
            embeddings_path = Path(config['paths'][f'path_to_{step}_data']) / f'{model_name}_{step}_{embeddings_name}'
            labels_path = Path(config['paths'][f'path_to_{step}_label']) / f'{model_name}_{step}_{labels_name}'

            # Load labels and convert to one-hot encoding
            labels = torch.tensor(
                np.load(labels_path).flatten(),  # Changed to np.load for consistency
                dtype=torch.long
            )
            labels = torch.nn.functional.one_hot(
                labels,
                num_classes=config['model_parameters']['num_classes']
            ).to(torch.float32)

            # Determine the embeddings shape for the current step
            embeddings_shape = config['data_loader_parameters'][f'embeddings_{step}_shape']
            embeddings_shape[0] = len(labels)

            # Load embeddings and reshape
            embeddings = torch.tensor(
                np.load(embeddings_path).reshape(embeddings_shape),
                dtype=torch.float32
            )

            # Create dataset and DataLoader
            dataset = CommentDataset(embeddings, labels)
            plot_label_distribution(dataset, step)

            if step == 'train':
                # Split the dataset into training and validation sets
                train_size = int(config['data_loader_parameters']['train_ratio'] * len(dataset))
                val_size = len(dataset) - train_size
                train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

                # Create DataLoaders for training and validation datasets
                train_dataloader = DataLoader(
                    train_dataset,
                    batch_size=config['data_loader_parameters']['batch_size'],
                    shuffle=config['data_loader_parameters']['shuffle'],
                    num_workers=config['data_loader_parameters']['num_workers'],
                    persistent_workers=config['data_loader_parameters']['persistent_workers']
                )
                val_dataloader = DataLoader(
                    val_dataset,
                    batch_size=config['data_loader_parameters']['batch_size'],
                    num_workers=config['data_loader_parameters']['num_workers'],
                    persistent_workers=config['data_loader_parameters']['persistent_workers']
                )
            else:
                # Create DataLoader for test dataset
                test_dataloader = DataLoader(
                    dataset,
                    batch_size=config['data_loader_parameters']['batch_size'],
                    num_workers=config['data_loader_parameters']['num_workers'],
                    persistent_workers=config['data_loader_parameters']['persistent_workers']
                )

        return train_dataloader, val_dataloader, test_dataloader


def plot_label_distribution(dataset: CommentDataset, step: str):
    """
    Plots the distribution of labels in the CommentDataset.

    Parameters:
    - dataset (CommentDataset): The dataset containing embeddings and labels.
    """
    # Extract labels from the dataset
    labels_array = np.array([dataset[i][1].argmax().item() for i in range(len(dataset))])
    # Count occurrences of each label
    unique_labels, counts = np.unique(labels_array, return_counts=True)
    label_counts = dict(zip(unique_labels, counts))

    # Print label counts
    print(f'label count for step {step} : {label_counts}')


class LSTMClassifier(L.LightningModule):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, num_classes: int):
        """
        Initializes the LSTMClassifier.

        Parameters:
        - input_dim (int): Dimension of the input features.
        - hidden_dim (int): Number of hidden units in the LSTM.
        - num_layers (int): Number of LSTM layers.
        - num_classes (int): Number of output classes.
        """
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)
        self.criterion = nn.BCEWithLogitsLoss() if num_classes == 1 else nn.CrossEntropyLoss()

        # Initialize tensors for predictions and targets
        self.predictions = torch.tensor([], dtype=torch.float32).to(self.device)
        self.targets = torch.tensor([], dtype=torch.float32).to(self.device)

        # Initialize appropriate F1 score metric
        self.metric = BinaryF1Score() if num_classes == 1 else MulticlassF1Score(num_classes=num_classes,
                                                                                 average='weighted')
        self.f1_score = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Parameters:
        - x (torch.Tensor): Input tensor.

        Returns:
        - torch.Tensor: Output logits.
        """
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # Take the last output from the LSTM
        return out

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

        y_hat = y_hat.to(self.predictions.device)
        y = y.to(self.targets.device)

        if y_hat.dim() > 1 and y_hat.size(1) > 1:
            # Multiclass classification
            y_hat = torch.argmax(y_hat, dim=1)
            y = torch.argmax(y, dim=1)
        else:
            # Binary classification (use threshold of 0.5)
            y_hat = torch.sigmoid(y_hat)
            y_hat = (y_hat > 0.5).float()
            y = y.float()

        self.metric.update(y_hat, y)
        self.log("Val_F1_score", self.metric.compute())
        self.log('learning_rate', self.trainer.optimizers[0].param_groups[0]['lr'])

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

        preds = pred.to(self.predictions.device)
        y = y.to(self.targets.device)

        # Concatenate predictions and targets
        self.predictions = torch.cat([self.predictions, preds])
        self.targets = torch.cat([self.targets, y])

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


class LogF1ScoreCallback(Callback):
    def __init__(self, model, model_name=None, hidden_dim=None, num_layers=None):
        """
        Initializes the LogF1ScoreCallback.

        Parameters:
        - model (LSTMClassifier): The model being evaluated.
        - model_name (str, optional): Name of the model (for logging/plotting purposes).
        - hidden_dim (int, optional): Number of hidden dimensions in the model (for logging/plotting).
        - num_layers (int, optional): Number of LSTM layers in the model (for logging/plotting).
        """
        super().__init__()
        self.model = model
        self.model_name = model_name
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

    def on_test_end(self, trainer, L):
        """
        Called at the end of the test phase.

        Computes and logs the F1 score, and optionally plots the confusion matrix.

        Parameters:
        - trainer (Trainer): The PyTorch Lightning trainer.
        - L (LightningModule): The Lightning model being tested.
        """
        # Convert logits to class predictions
        if self.model.predictions.dim() > 1 and self.model.predictions.size(1) > 1:
            # Multiclass classification
            predictions = torch.argmax(self.model.predictions, dim=1)
            targets = torch.argmax(self.model.targets, dim=1)
        else:
            # Binary classification
            predictions = (torch.sigmoid(self.model.predictions) > 0.5).long()
            targets = self.model.targets.long()

        # Ensure predictions and targets are on CPU for logging and plotting
        predictions_np = predictions.cpu().numpy()
        targets_np = targets.cpu().numpy()

        print(f'predictions from the mdoel : {self.model.predictions}')

        # Update metric and compute F1 score
        self.model.metric.update(predictions, targets)
        self.model.f1_score = self.model.metric.compute().item()
        print(f'Test F1 Score: {self.model.f1_score}')

        # Log the F1 score using the trainer's logger
        trainer.logger.log_metrics({"Test_F1_score": self.model.f1_score})
        plot_confusion_matrix(confusion_matrix(targets_np, predictions_np), self.model_name, self.hidden_dim,
                              self.num_layers, self.model.f1_score)


def plot_f1_score(dict_parameters: dict, model_name: str):
    """
    Plots a heatmap of F1 scores for various hidden dimensions and number of layers.

    Parameters:
    - dict_parameters (dict): Dictionary with keys as tuples (hidden_dim, num_layers)
                              and values as tuples (f1_score, predictions).
    - model_name (str): Name of the model for the plot title.
    """
    hidden_dims = []
    num_layers = []
    f1_scores = []

    # Extract hidden dimensions, number of layers, and F1 scores from the dictionary
    for (hidden_dim, num_layers_val), (f1_score, predictions, targets) in dict_parameters.items():
        hidden_dims.append(hidden_dim)
        num_layers.append(num_layers_val)
        f1_scores.append(f1_score.item() if isinstance(f1_score, torch.Tensor) else f1_score)

        # Create a DataFrame from the extracted data
    df = pd.DataFrame({
        'hidden_dim': hidden_dims,
        'num_layers': num_layers,
        'F1_score': f1_scores
    })

    # Pivot the DataFrame to format it for heatmap plotting
    df_pivot = df.pivot(index='hidden_dim', columns='num_layers', values='F1_score')

    # Plot heatmap using seaborn
    sns.heatmap(df_pivot, annot=True, cmap='viridis', fmt=".5f",
                cbar=True, vmin=df_pivot.min().min(), vmax=df_pivot.max().max())
    plt.title(f'F1 Score for {model_name} Model')
    plt.xlabel('Number of Layers')
    plt.ylabel('Hidden Dimensions')
    plt.show()


def plot_confusion_matrix(cm: np.ndarray, model_name: str, hidden_dim: int, num_layers: int, f1_score: float):
    """
    Plots a confusion matrix heatmap.

    Parameters:
    - cm (np.ndarray): Confusion matrix to plot.
    - model_name (str): Name of the model for the plot title.
    - hidden_dim (int): Hidden dimension used in the model.
    - num_layers (int): Number of layers used in the model.
    - f1_score (float): F1 score of the model, to display on the plot.
    """
    # Create a figure for the confusion matrix
    plt.figure(figsize=(8, 6))

    # Plot heatmap using seaborn
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='g', cbar=False)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')

    # Title of the plot, includes model details and F1 score
    plt.title(f'Confusion Matrix for {model_name}_{hidden_dim}_{num_layers}\n'
              f'with F1 Score: {f1_score:.4f}')

    # Display the plot
    plt.show()


def save_best_predictions(dict_parameters, model_name):
    """
    Saves the predictions and targets of the best performing models based on their F1 scores.

    Parameters:
    - dict_parameters (dict): Dictionary containing F1 scores, predictions, and targets.
    - model_name (str): Name of the model.

    Saves:
    - Numpy arrays of predictions and targets for each best performing model.
    """
    number_f1_score = config['save']['save_n_probability']

    # Sort the dictionary by F1 score in descending order
    dict_parameters = dict(sorted(dict_parameters.items(), key=lambda x: x[1][0], reverse=True))
    print(dict_parameters)
    # Save predictions for the top N models based on F1 score
    for i in range(number_f1_score):
        # Your code here for each iteration
        print(f"Iteration {i + 1}")
        key = list(dict_parameters.keys())[i]
        f1_score, predictions, targets = dict_parameters[key]

        print(f'F1 score: {f1_score:.4f}')
        print(f'Predictions: {predictions}')
        print(f'Targets: {targets}')

        # Generate a unique filename based on model name, hidden dimension, and number of layers
        name_save = f'{model_name}_{key[0]}_{key[1]}.npy'

        # Create directories to save predictions and targets if they don't exist
        path_predictions = Path(config['save']['path_to_save_predictions'])
        path_predictions.mkdir(parents=True, exist_ok=True)

        path_targets = Path(config['save']['path_to_save_targets'])
        path_targets.mkdir(parents=True, exist_ok=True)

        # Print information about saving process
        print(f'Saving predictions to {path_predictions / name_save}')
        print(f'Saving targets to {path_targets / name_save}')
        print(f'F1 score: {f1_score:.4f}')

        # Save predictions and targets as Numpy arrays
        np.save(path_predictions / name_save, predictions)
        np.save(path_targets / name_save, targets)


def train():
    """
    Trains LSTM models with different configurations specified in the configuration file.
    """
    for model_name in config['models_name']:
        print(f'Training model: {model_name}')
        dict_model = {}

        # Create directory to save models
        model_path = Path(config['paths']['path_to_model_directory'])
        model_path.mkdir(parents=True, exist_ok=True)

        # Load data using CommentDataset's data_load method
        train_dataloader, val_dataloader, test_dataloader = CommentDataset.data_load(model_name)

        # Iterate over different configurations of hidden dimensions and number of layers
        for hidden_dim in config['model_parameters']['train_hidden_dimensions']:
            for num_layers in config['model_parameters']['train_numbers_layers']:
                # Initialize LSTMClassifier model
                model = LSTMClassifier(
                    input_dim=config['model_parameters']['input_dim'],
                    hidden_dim=hidden_dim,
                    num_layers=num_layers,
                    num_classes=config['model_parameters']['num_classes']
                )

                # Initialize PyTorch Lightning Trainer
                trainer = L.Trainer(
                    log_every_n_steps=config['model_parameters']['log_every_n_steps'],
                    max_epochs=config['model_parameters']['max_epochs'],
                    logger=WandbLogger(),  # Logging with WandB
                    callbacks=[LogF1ScoreCallback(model, model_name, hidden_dim, num_layers)]
                )

                # Initialize WandB run
                wandb.init(
                    project=config['wandb_parameters']['wandbproject'],
                    name=f'{model_name}_{hidden_dim}_{num_layers}',
                    settings=wandb.Settings(quiet=True)
                )

                try:
                    # Train the model
                    trainer.fit(model, train_dataloader, val_dataloader)

                    # Test the model
                    trainer.test(model, test_dataloader)

                    # Retrieve F1 score from the trained model
                    f1_score = model.f1_score
                    print(f"F1 Score: {f1_score}")

                    # Store F1 score, predictions, and targets in dict_model
                    if f1_score is not None:
                        key = (hidden_dim, num_layers)
                        dict_model[key] = f1_score, model.predictions, model.targets
                    else:
                        print("F1 score was not computed.")

                except Exception as e:
                    print(f"Exception occurred during training: {e}")

                finally:
                    # Finish WandB run
                    wandb.finish()

                # Save trained model
                model_filename = f'{hidden_dim}_{num_layers}{config["other_parameters"]["model_extension"]}'
                path_to_save = model_path / model_name
                path_to_save.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), path_to_save / model_filename)

        # Plot F1 scores for all configurations
        plot_f1_score(dict_model, model_name)

        # Save predictions and targets for the best models
        save_best_predictions(dict_model, model_name)


if __name__ == '__main__':
    train()
