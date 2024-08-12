import lightning as L
import torch
import wandb
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import DataLoader

from Methods.Interface.Interface import LoadData, initialise_model
from config_loader import config
L.seed_everything(42)



def training(train_dataloader, val_dataloader, test_dataloader, device, model_name):

    for hidden_dim in config['model_parameters']['train_hidden_dimensions']:
        for num_layers in config['model_parameters']['train_numbers_layers']:
            params = {
                'hidden_dim': hidden_dim,
                'num_layers': num_layers,
                'model_name': model_name,
            }
            model = initialise_model('lstm', lstm_params=params)
            model.to(device)

            trainer = L.Trainer(
                max_epochs=config['model_parameters']['max_epochs'],
                logger=WandbLogger(),  # Logging with WandB
            )

            # Initialize WandB run
            wandb.init(
                project=config['wandb']['project'],
                name=f'{model_name}_{hidden_dim}_{num_layers}',
                settings=wandb.Settings(quiet=True)
            )

            try:

                trainer.fit(model, train_dataloader, val_dataloader)

                trainer.test(model, test_dataloader)

            except Exception as e:
                raise e

            finally:
                wandb.finish()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for model_name in config['model_names']:

        train_set, val_set, test_set = LoadData('lstm', model_name=model_name).__getattr__()

        train_dataloader = DataLoader(train_set, batch_size=config['batch_size'], shuffle=True)
        val_dataloader = DataLoader(val_set, batch_size=config['batch_size'], shuffle=False)
        test_dataloader = DataLoader(test_set, batch_size=config['batch_size'], shuffle=False)

        print(f'Data Loaded for {model_name} model')
        print(f'Let''s train the model')
        training(train_dataloader, val_dataloader, test_dataloader, device, model_name)

if __name__ == '__main__':
    main()
