"""
Training und Validierung des Modells.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from config import get_device, DATASET_DIR
from dataset import LoadDataset
from model import EmotionResNet


def train_epoch(model, device, train_loader, criterion, optimizer):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    train_loss = running_loss / len(train_loader)
    train_acc = (correct / total) * 100
    return train_loss, train_acc


def test_epoch(model, device, test_loader, criterion):
    model = model.to(device)
    model.eval()
    val_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    avg_val_loss = val_loss / len(test_loader)
    val_accuracy = (correct / total) * 100
    return avg_val_loss, val_accuracy


def plot_results(train_losses, val_losses, train_accuracies, val_accuracies):
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b', label='Training Loss')
    plt.plot(epochs, val_losses, 'r', label='Validation Loss')
    plt.title('Loss Curve (RAF-DB)')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, 'b', label='Training Accuracy')
    plt.plot(epochs, val_accuracies, 'r', label='Validation Accuracy')
    plt.title('Accuracy Curve (RAF-DB)')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()


def run_training(device, dataset, batch_size=32, learning_rate=0.001, num_epochs=20):
    train_dataset = LoadDataset(root=os.path.join(dataset, 'train'))
    val_dataset = LoadDataset(root=os.path.join(dataset, 'validation'))

    num_classes = len(train_dataset.dataset.classes)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)

    model = EmotionResNet(num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.0001)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0.00001)

    train_losses, train_accuracies = [], []
    val_losses, val_accuracies = [], []

    print('Started model training.')

    for epoch in range(num_epochs):
        if epoch % 5 == 0:
            print(f'Progress......{int(epoch / num_epochs * 100)}%')

        train_loss, train_accuracy = train_epoch(model, device, train_loader, criterion, optimizer)
        val_loss, val_accuracy = test_epoch(model, device, test_loader, criterion)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)

        scheduler.step()

    print('Training complete.')
    print(f"Final Validation Accuracy: {val_accuracies[-1]:.4f}, Final Validation Loss: {val_losses[-1]:.4f}")

    model_path = "trained_model_rafdb.pth"
    torch.save(model, model_path)
    print(f"Model saved to {model_path}.")

    plot_results(train_losses, val_losses, train_accuracies, val_accuracies)


if __name__ == '__main__':
    device = get_device()
    run_training(device, DATASET_DIR)
