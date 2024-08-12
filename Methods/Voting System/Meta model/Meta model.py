from collections import Counter

import lightning as L
import torch

from torch.utils.data import DataLoader

from Methods.Interface.Interface import LoadData, initialise_model, check_dataset, training_model
from config_loader import config
L.seed_everything(42)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_set, val_set, test_set = LoadData('metamodel', model_name='metamodel', on_approach='lstm').__getattr__()

    train_dataloader = DataLoader(train_set, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_set, batch_size=config['batch_size'], shuffle=False)
    test_dataloader = DataLoader(test_set, batch_size=config['batch_size'], shuffle=False)

    training_model(train_dataloader, val_dataloader, test_dataloader, device, 'metamodel')



if __name__ == '__main__':
    main()
