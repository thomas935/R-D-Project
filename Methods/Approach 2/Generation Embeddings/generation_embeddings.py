from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from Methods.Interface.Interface import LoadData, initialise_model, load_model, get_outputs
from config_loader import config


def generate_embeddings(model, dataloader, device, train, model_name):
    embeddings_list = []
    labels_list = []

    for batch in tqdm(dataloader, desc=f"Generating {model_name} embeddings"):
        outputs, label = get_outputs(batch, model, device)
        embeddings = outputs.last_hidden_state
        flattened_embeddings = embeddings.view(embeddings.size(0), -1)

        flattened_embeddings = flattened_embeddings.cpu().detach()
        label = label.cpu().detach()

        embeddings_list.append(flattened_embeddings)
        labels_list.append(label)

        if train:
            path_embeddings = Path(f"{config['path_to_content_root']}{config['save']}/Embeddings/Train")
            path_labels = Path(f"{config['path_to_content_root']}{config['save']}/Labels/Train")
        else:
            path_embeddings = Path(f"{config['path_to_content_root']}{config['save']}/Embeddings/Test")
            path_labels = Path(f"{config['path_to_content_root']}{config['save']}/Labels/Test")

        # Use mkdir(parents=True) to avoid errors if the directory already exists
        path_embeddings.mkdir(parents=True, exist_ok=True)
        path_labels.mkdir(parents=True, exist_ok=True)

        torch.save(embeddings_list, path_embeddings / f'{model_name}_embeddings.npy')
        torch.save(labels_list, path_labels / f'{model_name}_labels.npy')


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(config)
    percentage_of_data = config['percentage_of_data']

    for model_name in config['model_names']:
        print(f'Loading data for model {model_name}')
        train_set, test_set = LoadData('transformer', model_name, percentage_of_data=percentage_of_data).__getattr__()

        train_dataloader = DataLoader(train_set, batch_size=config['batch_size'], shuffle=True)
        test_dataloader = DataLoader(test_set, batch_size=config['batch_size'], shuffle=False)

        model = load_model(model_name, device, embedding=True)
        model.to(device)

        generate_embeddings(model, train_dataloader, device, True, model_name)
        generate_embeddings(model, test_dataloader, device, False, model_name)


if __name__ == '__main__':
    main()