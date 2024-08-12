# R-D-Project

## Overview

This project implements a research and development system using various machine learning models, including LSTM and a
meta-model approach. The project is built using Python and leverages libraries such as PyTorch, Lightning, and WandB for
training and logging.

## Project Structure

- `Methods/Voting System/`
    - `VotingSystem.py`: Contains functions for different voting mechanisms and evaluation metrics.
- `Methods/Approach 2/Training LSTM/`
    - `train_lstm.py`: Script for training LSTM models.
- `Methods/Voting System/Meta model/`
    - `Meta model.py`: Script for training meta-models.
- `Methods/Interface/`
    - `Interface.py`: Contains utility functions for loading data and initializing models.
- `config_loader.py`: Loads configuration settings for the project.

## Requirements

- Python 3.x
- pip

## Installation

1. Clone the repository:
   ```sh
   git clone <repository-url>
   cd <repository-directory>
   ```

2. Install the required packages:
   ```sh
   pip install -r requirements.txt
   ```

## Configuration

Update the `config_loader.py` file with the appropriate paths and parameters for your environment.

## Usage

### Training LSTM Models

To train LSTM models, run the following command:

```sh
python Methods/Approach\ 2/Training\ LSTM/train_lstm.py
```

### Training Meta-Models

To train meta-models, run the following command:

```sh
python Methods/Voting\ System/Meta\ model/Meta\ model.py
```

### Voting System

To evaluate the voting system, run the following command:

```sh
python Methods/Voting\ System/VotingSystem.py
```

## Functions

### VotingSystem.py

- `mean_vote(predictions, labels)`: Computes the mean vote.
- `majority_vote(predictions, labels)`: Computes the majority vote.
- `sum_square_vote(predictions, labels)`: Computes the sum square vote.
- `vote(predictions, labels)`: Aggregates all voting methods.
- `model_f1_score(labels, application)`: Computes the F1 score for a given model.
- `check_labels_predictions(predictions, labels)`: Checks the consistency of predictions and labels.

### train_lstm.py

- `training(train_dataloader, val_dataloader, test_dataloader, device, model_name)`: Trains the LSTM model.

### Meta model.py

- `check_custom_dataset(custom_dataset)`: Checks the custom dataset.
- `trainings(train_dataloader, val_dataloader, test_dataloader, device, model_name)`: Trains the meta-model.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

## Acknowledgements

- [PyTorch](https://pytorch.org/)
- [Lightning](https://www.pytorchlightning.ai/)
- [WandB](https://wandb.ai/)

For any questions or issues, please open an issue on the repository.