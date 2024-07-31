from config_loader import config
from Methods.Interface.Interface import (CustomTransformerModel, CustomDataset,
    LoadData)

def main():

    for model_name in config['model_names']:
        # Load data
        print(f'Loading data for model {model_name}')
        train_set, test_set = LoadData('transformer', model_name).__getitem__()


if __name__ == '__main__':
    main()
