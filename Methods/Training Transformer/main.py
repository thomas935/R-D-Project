import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from Methods.Interface.Interface import (CustomTransformerModel, check_dataloader, check_dataset,
                                         LoadData, initialise_model)
from config_loader import config


def train(model, train_dataloader, device):
    num_epochs = config['num_epochs']
    learning_rate = float(config['learning_rate'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    model.train()  # Set model to training mode

    losses = []

    for epoch in range(num_epochs):

        interval = len(train_dataloader) // 5

        epoch_loss_values = []
        loss_values = []

        for batch_idx, batch in enumerate(tqdm(train_dataloader, desc=f"Training Epoch {epoch + 1}/{num_epochs}")):
            # Prepare batch
            ids = batch['ids'].to(device, dtype=torch.long)
            mask = batch['mask'].to(device, dtype=torch.long)
            labels = batch['labels'].to(device, dtype=torch.float)

            # Zero the gradients
            model.zero_grad()

            # Forward pass
            logits = model(ids, mask)

            loss = F.binary_cross_entropy_with_logits(logits, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            epoch_loss_values.append(loss.item())

            if (batch_idx + 1) % interval == 0:
                batch_loss = sum(epoch_loss_values) / len(
                    epoch_loss_values)  # Calculate average loss from epoch_loss_values
                print(f'Epoch: {epoch}, Batch: {batch_idx + 1}, Loss: {batch_loss:.4f}')
                loss_values.append(batch_loss)
                epoch_loss_values = []

        # Print epoch statistics
        avg_loss = sum(loss_values) / len(loss_values)
        print(f'Epoch: {epoch}, Average Loss: {avg_loss:.4f}')

    print("Training complete!")


def test(model: CustomTransformerModel, test_dataloader: DataLoader, device: torch.device):
    model.eval()
    initial_preds = []
    initial_labels = []
    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Initial Predictions on Test Set"):
            ids = batch['ids'].to(device, dtype=torch.long)
            mask = batch['mask'].to(device, dtype=torch.long)
            labels = batch['labels'].to(device, dtype=torch.long)
            labels = torch.argmax(labels, dim=1)
            logits = model(ids, mask)
            preds = torch.argmax(logits, dim=1)
            initial_preds.extend(preds.cpu().numpy())
            initial_labels.extend(labels.cpu().numpy())

    print(initial_preds)
    print(initial_labels)
    f1 = f1_score(initial_labels, initial_preds, average='weighted')
    return f1


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    percentage_of_data = config['percentage_of_data']
    for model_name in config['model_names']:
        # Load data
        print(f'Loading data for model {model_name}')
        train_set, test_set = LoadData('transformer', model_name, percentage_of_data).__getattr__()
        check_dataset(test_set)

        train_dataloader = DataLoader(train_set, batch_size=config['batch_size'], shuffle=True)
        test_dataloader = DataLoader(test_set, batch_size=config['batch_size'], shuffle=False)
        check_dataloader(test_dataloader)

        # Check for the initial F1 score without training
        model = initialise_model(model_name, device)
        f1 = test(model, test_dataloader, device)
        print(f'Initial F1 score for model {model_name}: {f1}')

        # Train the model
        train(model, train_dataloader, device)

        f1 = test(model, test_dataloader, device)
        print(f'Final F1 score for model {model_name}: {f1}')


if __name__ == '__main__':
    main()
