import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader

from tqdm import tqdm
import wandb


wandb.init(project="maps_resnet_classification", entity="hails", name="resnet34_maps")

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
batch_size = 32
epochs = 10
learning_rate = 0.001
train_data_dir = '/media/hail/HDD/style_transfer/results/maps_cyclegan/test_latest/images/'
test_data_dir = '/media/hail/HDD/style_transfer/datasets/maps/val_dataset'

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

train_dataset = datasets.ImageFolder(train_data_dir, transform=transform)
test_dataset = datasets.ImageFolder(test_data_dir, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

class ResNetClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super(ResNetClassifier, self).__init__()
        self.model = models.resnet34(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

model = ResNetClassifier(num_classes=2).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

wandb.watch(model, criterion, log="all", log_freq=10)

def train(model, train_loader, criterion, optimizer, epoch):
    model.train()
    running_loss = 0.0
    train_loader_tqdm = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{epochs}] Training", leave=False)
    for images, labels in train_loader_tqdm:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        train_loader_tqdm.set_postfix(loss=(running_loss / len(train_loader)))

    avg_loss = running_loss / len(train_loader)
    wandb.log({"Train Loss": avg_loss, "Epoch": epoch + 1})
    print(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}")


def evaluate(model, test_loader, criterion, epoch):
    model.eval()
    correct = 0
    total = 0
    running_loss = 0.0
    test_loader_tqdm = tqdm(test_loader, desc="Evaluating", leave=False)
    with torch.no_grad():
        for images, labels in test_loader_tqdm:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            # Calculate loss
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            # Calculate accuracy
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            test_loader_tqdm.set_postfix(accuracy=(100 * correct / total), loss=(running_loss / len(test_loader)))

    # Calculate average loss and accuracy
    avg_loss = running_loss / len(test_loader)
    accuracy = 100 * correct / total

    # Log metrics to wandb
    wandb.log({"Test Loss": avg_loss, "Test Accuracy": accuracy, "Epoch": epoch + 1})

    print(f"Test Loss: {avg_loss:.4f}, Test Accuracy: {accuracy:.2f}%")
    return avg_loss, accuracy

for epoch in range(epochs):
    train(model, train_loader, criterion, optimizer, epoch)
    test_loss, test_accuracy = evaluate(model, test_loader, criterion, epoch)

torch.save(model.state_dict(), 'resnet_xray_classification.pth')
wandb.save('resnet_xray_classification.pth')
print("Model saved successfully!")
