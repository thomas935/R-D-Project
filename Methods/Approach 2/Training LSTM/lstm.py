import torch
from torch.utils.data import DataLoader

from Methods.Interface.Interface import LoadData, load_model, check_dataset, check_dataloader
from config_loader import config


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for model_name in config['model_names']:
        print(f'Loading data for model {model_name}')

        train_set, test_set = LoadData('lstm', model_name).__getattr__()
        check_dataset(train_set)

        train_dataloader = DataLoader(train_set, batch_size=config['batch_size'], shuffle=True)
        test_dataloader = DataLoader(test_set, batch_size=config['batch_size'], shuffle=False)
        check_dataloader(train_dataloader)

        model = load_model(model_name,'lstm', embedding=True)
        model.to(device)


if __name__ == '__main__':
    main()
