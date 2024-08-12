import lightning as L
import torch
import wandb
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import DataLoader

from Methods.Interface.Interface import LoadData, initialise_model, training_model
from config_loader import config
L.seed_everything(42)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for model_name in config['model_names']:

        train_set, val_set, test_set = LoadData('lstm', model_name=model_name).__getattr__()

        train_dataloader = DataLoader(train_set, batch_size=config['batch_size'], shuffle=True)
        val_dataloader = DataLoader(val_set, batch_size=config['batch_size'], shuffle=False)
        test_dataloader = DataLoader(test_set, batch_size=config['batch_size'], shuffle=False)

        print(f'Data Loaded for {model_name} model')
        print(f'Let''s train the model')
        training_model(train_dataloader, val_dataloader, test_dataloader, device, 'lstm', model_name)

if __name__ == '__main__':
    main()
