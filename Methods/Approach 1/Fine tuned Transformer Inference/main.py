import numpy as np
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm

from Methods.Interface.Interface import LoadData, load_model, get_outputs
from config_loader import config
from sklearn.metrics import f1_score
#import f1_score from sklearn





def predict(model, dataloader, device, model_name):
    predictions = []
    labels = []

    for batch in tqdm(dataloader, desc=f"Generating {model_name} Predictions"):
        outputs, label = get_outputs(batch, model, device)
        predictions.append(outputs)
        labels.append(label)
    predictions = torch.cat(predictions).cpu().detach().numpy()
    print(predictions)
    predicted_labels = np.argmax(predictions, axis=1)
    print(predicted_labels)
    labels = np.argmax(torch.cat(labels).cpu().detach().numpy(), axis=1)
    print(labels)
    f1 = f1_score(labels, predicted_labels, average='weighted')
    print(f'F1 score for model {model_name}: {f1}')

    path_to_predictions = Path(f"{config['path_to_content_root']}{config['save']}/Predictions")
    path_to_predictions.mkdir(parents=True, exist_ok=True)

    path_to_labels = Path(f"{config['path_to_content_root']}{config['save']}/Labels")
    path_to_labels.mkdir(parents=True, exist_ok=True)

    torch.save(predictions, path_to_predictions / f'{model_name}_predictions.npy')
    torch.save(labels, path_to_labels / f'{model_name}_labels.npy')


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    percentage_of_data = config['percentage_of_data']

    for model_name in config['model_names']:
        print(f'Loading data for model {model_name}')
        test_set = LoadData('transformer', model_name, percentage_of_data=percentage_of_data).__getattr__('test_set')

        test_dataloader = DataLoader(test_set, batch_size=config['batch_size'], shuffle=False)

        model = load_model(model_name, device, embedding=False)
        model.to(device)

        predict(model, test_dataloader, device, model_name)

if __name__ == '__main__':
    main()
