from pathlib import Path

import lightning as L
import numpy as np
import pandas as pd
import torch
import wandb
from lightning.pytorch.callbacks import EarlyStopping, Callback
from lightning.pytorch.loggers import WandbLogger
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torcheval.metrics import BinaryF1Score

L.seed_everything(42)
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from config_loader import config
import seaborn as sns
from sklearn.metrics import confusion_matrix


class MetaModel(L.LightningModule):

    def __init__(self, input_dimension, output_dimension, hidden_size, num_layers):
        super(MetaModel, self).__init__()

        layers = []
        # Input layer
        layers.append(nn.Linear(config['input_dim'], hidden_size))
        layers.append(nn.ReLU())

        # Hidden layers
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())

        # Output layer
        layers.append(nn.Linear(hidden_size, config['output_dim']))

        # For binary classification
        if config['output_dim'] == 1:
            layers.append(nn.Sigmoid())

        self.model = nn.Sequential(*layers)
        self.criterion = nn.BCELoss()
        self.metric = BinaryF1Score()

        self.predictions = torch.tensor([], dtype=torch.float32).to(self.device)
        self.targets = torch.tensor([], dtype=torch.float32).to(self.device)
        self.f1_score = 0

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        y_hat = y_hat.squeeze()
        loss = self.criterion(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):

        x, y = batch

        y_hat = self(x)
        y_hat = y_hat.squeeze()

        loss = self.criterion(y_hat, y)
        self.log("val_loss", loss)

        y_hat = y_hat.to(self.predictions.device)
        y = y.to(self.targets.device)

        self.metric.update(y_hat.squeeze(), y.squeeze())
        self.log("Val_F1_score", self.metric.compute())
        self.log('learning_rate', self.trainer.optimizers[0].param_groups[0]['lr'])

    def on_validation_end(self):
        self.metric.reset()

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        y_hat = y_hat.squeeze()

        loss = self.criterion(y_hat, y)
        self.log("test_loss", loss)

        y_hat = y_hat.to(self.predictions.device)
        y = y.to(self.targets.device)

        self.predictions = torch.cat([self.predictions, y_hat])
        self.targets = torch.cat([self.targets, y])

    def on_test_end(self):

        print(f'This is test end : ')
        print(f'Predictions : {self.predictions}')
        print(f'Targets : {self.targets}')
        print(f'F1_score : {self.metric.compute()}')

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=float(config['learning_rate']))
        scheduler = ReduceLROnPlateau(optimizer,
                                      mode=config['scheduler']['mode'],
                                      factor=config['scheduler']['factor'],
                                      patience=config['scheduler']['patience_schedular'],
                                      threshold=config['scheduler']['threshold_schedular'])
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
                'frequency': 1,
                'monitor': 'val_loss',
                'strict': True,
            }
        }


class CommentDataset(Dataset):
    def __init__(self, predictions, targets):
        self.predictions = torch.tensor(predictions, dtype=torch.float32)  # Convert to tensor
        self.targets = torch.tensor(targets, dtype=torch.float32)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return {
            'prediction': self.predictions[idx],
            'target': self.targets[idx]
        }

    def __str__(self):
        return f"Dataset: Predictions shape={self.predictions.shape}, Targets shape={self.targets.shape}"


def plot_target_distribution(y_train, y_val, y_test):
    """
    Plot the distribution of target classes as percentages and print counts for each dataset.

    Parameters:
    - y_train (numpy array): Target variable for the training set.
    - y_val (numpy array): Target variable for the validation set.
    - y_test (numpy array): Target variable for the test set.
    """

    def count_labels(targets):
        unique, counts = np.unique(targets, return_counts=True)
        label_counts = dict(zip(unique, counts))
        label_0_count = label_counts.get(0)
        label_1_count = label_counts.get(1)
        return label_0_count, label_1_count

    # Calculate total counts for each dataset
    total_train = len(y_train)
    total_val = len(y_val)
    total_test = len(y_test)

    # Count labels for each array
    train_label_0_count, train_label_1_count = count_labels(y_train)
    val_label_0_count, val_label_1_count = count_labels(y_val)
    test_label_0_count, test_label_1_count = count_labels(y_test)

    print(f"Train - Label 0 count: {train_label_0_count}, Label 1 count: {train_label_1_count}")
    print(f"Validation - Label 0 count: {val_label_0_count}, Label 1 count: {val_label_1_count}")
    print(f"Test - Label 0 count: {test_label_0_count}, Label 1 count: {test_label_1_count}")

    # Calculate percentages for train, validation, and test sets
    percent_train_0 = (train_label_0_count / total_train) * 100
    percent_train_1 = (train_label_1_count / total_train) * 100

    percent_val_0 = (val_label_0_count / total_val) * 100
    percent_val_1 = (val_label_1_count / total_val) * 100

    percent_test_0 = (test_label_0_count / total_test) * 100
    percent_test_1 = (test_label_1_count / total_test) * 100

    # Plotting the distributions
    plt.figure(figsize=(18, 6))

    plt.subplot(1, 3, 1)
    plt.bar([0, 1], [percent_train_0, percent_train_1], color=['skyblue', 'lightcoral'])
    plt.title('Training Set')
    plt.xlabel('Target Classes')
    plt.ylabel('Percentage (%)')
    plt.xticks([0, 1], ['Class 0', 'Class 1'])
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.subplot(1, 3, 2)
    plt.bar([0, 1], [percent_val_0, percent_val_1], color=['lightgreen', 'lightcoral'])
    plt.title('Validation Set')
    plt.xlabel('Target Classes')
    plt.ylabel('Percentage (%)')
    plt.xticks([0, 1], ['Class 0', 'Class 1'])
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.subplot(1, 3, 3)
    plt.bar([0, 1], [percent_test_0, percent_test_1], color=['salmon', 'lightcoral'])
    plt.title('Test Set')
    plt.xlabel('Target Classes')
    plt.ylabel('Percentage (%)')
    plt.xticks([0, 1], ['Class 0', 'Class 1'])
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()


def concatenate_probabilities(directory):
    arr = []
    for file_path in sorted(directory.iterdir()):
        temp_arr = np.load(file_path)
        arr.append(temp_arr)
    arr = np.hstack(arr)

    return arr


def get_targets(directory):
    file_name = 'Target (1).npy'
    arr = np.load(directory / file_name)
    # arr = [np.argmax(pair) for pair in arr]
    print(f'Loaded targets from {file_name}')
    print(f'First 10 targets: {arr[:10]}')
    return arr


def create_dataloader():
    probabilities_directory = Path(config['paths']['predictions'])
    probabilities = concatenate_probabilities(probabilities_directory)

    targets_directory = Path(config['paths']['targets'])
    targets = get_targets(targets_directory)

    print(probabilities)
    print(targets)

    X_train_val, X_test, y_train_val, y_test = train_test_split(probabilities, targets, test_size=0.2,
                                                                random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.125, random_state=42)
    print("Training set size:", X_train.shape)
    print("Validation set size:", X_val.shape)
    print("Test set size:", X_test.shape)

    # Plot target distribution
    plot_target_distribution(y_train, y_val, y_test)

    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                  torch.tensor(y_train, dtype=torch.float32))
    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32))
    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32),
                                 torch.tensor(y_test, dtype=torch.float32))

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=config['shuffle'],
                              num_workers=config['num_workers'], persistent_workers=config['persistent_workers'])
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False,
                            num_workers=config['num_workers'], persistent_workers=config['persistent_workers'])
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False,
                             num_workers=config['num_workers'], persistent_workers=config['persistent_workers'])

    return train_loader, val_loader, test_loader


class LogF1ScoreCallback(Callback):
    def __init__(self, model, model_name=None, hidden_dim=None, num_layers=None):
        super().__init__()
        self.model = model
        self.model_name = model_name
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.f1_score = 0

    def on_test_end(self, trainer, L):
        self.model.metric.update(self.model.predictions, self.model.targets)
        self.model.f1_score = self.model.metric.compute()
        self.f1_score = self.model.metric.compute()
        print(f'Test F1 Score: {self.f1_score}')
        trainer.logger.log_metrics({"Test_F1_score": self.f1_score})

        print(f'dimension prediction {self.model.predictions.dim()}')

        if self.model.predictions.dim() > 1 and self.model.predictions.size(1) > 1:
            # Multiclass classification
            predictions = torch.argmax(self.model.predictions, dim=1)
            targets = torch.argmax(self.model.targets, dim=1)
        else:
            # Binary classification
            predictions = (self.model.predictions > 0.5).long()
            targets = self.model.targets.long()

        print(f'Predictions: {predictions}')
        print(f'Targets: {targets}')

        # Ensure predictions and targets are on CPU for logging and plotting
        predictions_np = predictions.cpu().numpy()
        targets_np = targets.cpu().numpy()

        print(f'Printing predictions and targets before confusion matrix')
        print(f'predictions : {predictions_np}')
        print(f'Targets : {targets_np}')

        plot_confusion_matrix(
            confusion_matrix(targets_np, predictions_np),
            self.model_name, self.hidden_dim,
            self.num_layers, self.f1_score
        )


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


def plot_f1_score(dict_parameters: dict, model_name: str = 'MetaModel'):
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

    print(dict)
    # Extract hidden dimensions, number of layers, and F1 scores from the dictionary
    for (hidden_dim, num_layers_val), (f1_score) in dict_parameters.items():
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


def main():
    train_loader, val_loader, test_loader = create_dataloader()
    wandb.login(key='dc45808b5077bb461397335e052f69badaad2620')
    dict_model = {}

    for hidd_size in config['hidden_size']:
        for num_layer in config['num_layers']:
            try:
                metamodel = MetaModel(
                    input_dimension=config['input_dim'],
                    output_dimension=config['output_dim'],
                    hidden_size=hidd_size,
                    num_layers=num_layer)

                early_stopping_callback = EarlyStopping(
                    monitor='val_loss',  # or any other metric you want to monitor
                    patience=config['scheduler']['patience_early_stopping'],  # number of epochs with no improvement
                    verbose=False,
                    mode=config['scheduler']['mode'],
                    min_delta=config['scheduler']['threshold_early_stopping']
                )

                trainer = L.Trainer(
                    max_epochs=config['max_epochs'],
                    logger=WandbLogger(),
                    callbacks=[
                        early_stopping_callback,
                        LogF1ScoreCallback(metamodel, model_name='MetaModel', hidden_dim=hidd_size,
                                           num_layers=num_layer)
                    ]
                )

                # Initialize wandb
                wandb.init(
                    name=f'num_layer_{num_layer}_h_size_{hidd_size}',
                    project=config['wandb']['project'],
                    settings=wandb.Settings(quiet=True)
                )

                trainer.fit(metamodel, train_loader, val_loader)
                trainer.test(metamodel, test_loader)

                f1_score = metamodel.f1_score
                print(f"F1 score for hidden size {hidd_size}, num layers {num_layer}: {f1_score}")
                # Store F1 score, predictions, and targets in dict_model
                if f1_score is not None:
                    key = (hidd_size, num_layer)
                    dict_model[key] = f1_score
                else:
                    print("F1 score was not computed.")
            except Exception as e:
                print(e)
            finally:
                wandb.finish()

    # Plot F1 scores for all configurations
    plot_f1_score(dict_model)


if __name__ == '__main__':
    main()
