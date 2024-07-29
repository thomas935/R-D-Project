import logging
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import cuda
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from transformers import PreTrainedTokenizer
from transformers import RobertaModel

device = 'cuda' if cuda.is_available() else 'cpu'

from config_loader import config

# Set logging level to ERROR to suppress the warnings
logging.getLogger("transformers").setLevel(logging.ERROR)


class RobertaClass(nn.Module):
    """
    PyTorch module for a text classification model using Roberta.

    This model uses a pre-trained RoBERTa model followed by a linear classification layer.
    The last hidden state from RoBERTa is processed through a dropout and a pre-classifier layer
    before being passed to the final classifier layer.

    Attributes:
    - l1 (RobertaModel): Pre-trained RoBERTa model.
    - pre_classifier (nn.Linear): Linear layer for intermediate processing.
    - dropout (nn.Dropout): Dropout layer for regularization.
    - classifier (nn.Linear): Linear layer for final classification.

    Parameters:
    - input_ids (torch.Tensor): Input IDs tensor of shape (batch_size, sequence_length).
    - attention_mask (torch.Tensor): Attention mask tensor of shape (batch_size, sequence_length).

    Returns:
    - torch.Tensor: Output logits of shape (batch_size, num_classes).
    """

    def __init__(self):
        super(RobertaClass, self).__init__()
        self.l1 = RobertaModel.from_pretrained("roberta-base")
        self.pre_classifier = torch.nn.Linear(768, 768)
        self.dropout = torch.nn.Dropout(0.3)
        self.classifier = torch.nn.Linear(768, 2)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the model.

        Processes the input IDs and attention mask through RoBERTa, applies a pre-classifier and dropout,
        and outputs the final logits.

        Parameters:
        - input_ids (torch.Tensor): Input IDs tensor of shape (batch_size, sequence_length).
        - attention_mask (torch.Tensor): Attention mask tensor of shape (batch_size, sequence_length).

        Returns:
        - torch.Tensor: Logits tensor of shape (batch_size, num_classes).
        """
        # Get the last hidden state from RoBERTa
        outputs = self.l1(input_ids=input_ids, attention_mask=attention_mask)

        return outputs


# Custom model class
class CustomModel(nn.Module):
    def __init__(self, tokenizer_name, num_labels):
        super(CustomModel, self).__init__()
        self.base_model = AutoModel.from_pretrained(tokenizer_name, output_hidden_states=True)
        self.pre_classifier = torch.nn.Linear(self.base_model.config.hidden_size, 768)
        self.dropout = torch.nn.Dropout(0.3)
        self.classifier = nn.Linear(768, num_labels)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

        return outputs


class TextDataset(Dataset):
    """
        Dataset class for text data to be used with a PyTorch DataLoader.

        This class handles tokenization of text and preparation of input tensors for a model,
        including input IDs, attention masks, and token type IDs.

        Attributes:
        - data (pd.DataFrame): The DataFrame containing the text and labels.
        - tokenizer (PreTrainedTokenizer): The tokenizer used to encode the text.
        - max_len (int): The maximum length for tokenization.

        Parameters:
        - dataframe (pd.DataFrame): DataFrame containing the 'text' and 'sexist_label' columns.
        - tokenizer (PreTrainedTokenizer): Tokenizer for encoding the text.
        - max_len (int): Maximum length for the tokenized sequences.
        """

    def __init__(self, dataframe: pd.DataFrame, tokenizer: PreTrainedTokenizer, max_len: int):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.text = dataframe['text']
        self.labels = dataframe['sexist_label']
        self.max_len = max_len

    def __str__(self) -> str:
        return str(self.data)

    def __len__(self) -> int:
        return len(self.text)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Retrieves a tokenized sample from the dataset at the specified index.

        Parameters:
        - idx (int): The index of the sample to retrieve.

        Returns:
        Dict[str, Any]: A dictionary containing:
            - 'ids': Tensor of input IDs.
            - 'mask': Tensor of attention mask.
            - 'label': Tensor of the label.
            - 'token_type_ids': Tensor of token type IDs.
            - 'text': The original text.
        """
        # Get the text at the specified index
        text: str = str(self.text[idx])
        # Normalize whitespace in the text
        text = " ".join(text.split())

        # Tokenize the text
        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_token_type_ids=True
        )

        # Extract tokenization outputs
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs['token_type_ids']

        # Create a dictionary of tensors
        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'label': torch.tensor(self.labels[idx], dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'text': text
        }


def load_tokenizer(tokenizer_name):
    return AutoTokenizer.from_pretrained(tokenizer_name)


def load_model(model_name, model_path, tokenizer_name):
    if model_name == 'roberta':
        model = RobertaClass()
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)
    else:
        num_labels = 2
        model = CustomModel(tokenizer_name, num_labels)
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def generate_embeddings(df: pd.DataFrame, model_name: str, path: Path, train: bool) -> None:
    """
    Generates and saves embeddings for text data using a specified transformer model.

    This function tokenizes the text data, passes it through a transformer model to generate embeddings,
    and saves the resulting embeddings and labels to disk. Embeddings are saved as numpy arrays.

    Parameters:
    - df (pd.DataFrame): DataFrame containing the text data and labels.
    - model_name (str): Name of the model used for generating embeddings.
    - path (Path): Path to the directory where embeddings and labels will be saved.
    - train (bool): Indicates whether the data is training data or test data. Used to determine save location.

    Returns:
    - None
    """

    # Load tokenizer and model based on the configuration
    tokenizer = load_tokenizer(config['models'][model_name]['model_name'])
    model_path = Path(f'{config["models"]["path_model"]}/{model_name}.pth')
    print(model_path)

    model = load_model(model_name, model_path, config['models'][model_name]['model_name'])

    # Configuration parameters
    max_len = config['tokenizer_max_length']
    params = {
        'batch_size': config['params']['batch_size'],
        'shuffle': train,
        'num_workers': config['params']['num_workers']
    }

    # Create a dataset and dataloader
    dataset = TextDataset(df, tokenizer, max_len=max_len)
    dataloader = DataLoader(dataset, **params)

    embeddings_dict: Dict[tuple, Any] = {}

    for data in tqdm(dataloader, desc=f"Generating {model_name} embeddings"):
        try:
            texts = data['text']
            text_tuple = tuple(texts)
            labels = data['label']

            ids = data['ids'].to(device, dtype=torch.long)
            mask = data['mask'].to(device, dtype=torch.long)

            with torch.no_grad():
                outputs = model(input_ids=ids, attention_mask=mask)

            embeddings = outputs.last_hidden_state
            flattened_embeddings = embeddings.view(embeddings.size(0), -1)

            flattened_embeddings = flattened_embeddings.cpu().detach()

            embeddings_label_tuple = (flattened_embeddings, labels)
            embeddings_dict[text_tuple] = embeddings_label_tuple

            del embeddings, flattened_embeddings, outputs

        except RuntimeError as e:
            print(f"Runtime error: {e}")
            torch.cuda.empty_cache()

        torch.cuda.empty_cache()

    # Flatten the dictionary to map each text to its embedding and label
    flattened_embeddings_dict = {
        text: (embedding, label_elem)
        for texts_tuple, (embedding, label) in embeddings_dict.items()
        for text, embedding, label_elem in zip(texts_tuple, embedding, label)
    }

    embeddings_list = []
    labels_list = []

    for text, (embedding, label) in flattened_embeddings_dict.items():
        labels_list.append(label.cpu().detach().item())
        embeddings_list.append(embedding.cpu().detach().numpy())

    print(f"Embeddings shape: {np.array(embeddings_list).shape}")
    print(f"Labels shape: {np.array(labels_list).shape}")

    # Determine the file save paths based on whether the data is training or test data
    embeddings_save_name = config['save']['embeddings']['embeddings_save']
    labels_save_name = config['save']['labels']['labels_save']

    if train:
        path_embeddings = path / 'Embeddings' / 'Train'
        path_labels = path / 'Labels' / 'Train'
    else:
        path_embeddings = path / 'Embeddings' / 'Test'
        path_labels = path / 'Labels' / 'Test'

    path_embeddings.mkdir(parents=True, exist_ok=True)
    path_labels.mkdir(parents=True, exist_ok=True)

    # Save the embeddings and labels to disk
    np.save(path_embeddings / f'{model_name}_{"train" if train else "test"}_{embeddings_save_name}', embeddings_list)
    np.save(path_labels / f'{model_name}_{"train" if train else "test"}_{labels_save_name}', labels_list)


def text_formation() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Prepares training and testing datasets by sampling a fraction of the data.

    This function reads the training and testing datasets from specified paths, samples a small fraction of the data,
    and resets their indices. It is useful for testing purposes or to reduce dataset size for quicker iterations.

    Returns:
    - Tuple[pd.DataFrame, pd.DataFrame]: A tuple cont
    aining the sampled training and testing DataFrames.
    """
    # Get paths and filenames from the configuration
    path_train = Path(config['path_train_data'])
    if not path_train.exists():
        raise FileNotFoundError(f"Path {path_train} does not exist")
    name_train = config['name_train_data']

    path_test = Path(config['path_test_data'])
    if not path_test.exists():
        raise FileNotFoundError(f"Path {path_test} does not exist")
    name_test = config['name_test_data']

    # Load the test data
    test_df = pd.read_csv(path_test / name_test)

    # Load the train data
    train_df = pd.read_csv(path_train / name_train)

    '''# Sample 1% of the training and testing data
    train_df = train_df.sample(frac=0.1, random_state=42)
    test_df = test_df.sample(frac=0.1, random_state=42)

    # Reset indices
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)'''

    return train_df, test_df


def main() -> None:
    """
    Main function to prepare datasets and generate embeddings for specified models.

    This function prepares the training and testing datasets by calling `text_formation`. It then iterates over the
    models specified in the configuration, generates embeddings for both the training and testing datasets, and saves
    them to the specified directory.

    Returns:
    - None
    """
    # Prepare training and testing datasets
    train_df, test_df = text_formation()

    # Create the directory for saving embeddings if it doesn't exist
    path = Path(config['save']['path'])
    path.mkdir(parents=True, exist_ok=True)

    # Generate embeddings for each model specified in the configuration
    for model in config['models_name']:
        print(f"Generating embeddings for {model}")

        # Generate and save train embeddings
        print(f"Generating train embeddings for {model}")
        generate_embeddings(train_df, model, path, train=True)

        # Generate and save test embeddings
        print(f"Generating test embeddings for {model}")
        generate_embeddings(test_df, model, path, train=False)


if __name__ == '__main__':
    main()
