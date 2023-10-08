import torch.nn as nn
import torchvision.models as models
import torch
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
from torchvision.models.resnet import Wide_ResNet50_2_Weights
import config

print("init..")

print("Configuration Setup...")

# Configuration
IDX_CLASS_LABELS = {
    0: 'AnnualCrop',
    1: 'Forest', 
    2: 'HerbaceousVegetation',
    3: 'Highway',
    4: 'Industrial',
    5: 'Pasture',
    6: 'PermanentCrop',
    7: 'Residential',
    8: 'River',
    9: 'SeaLake'
}
MODEL_PATH = r'model path if you need to fine tune it'
NUM_CLASSES = 10
train_dir = r"training data here"

# Data Loading

print("Setting up Data Loading...")

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
dataset = ImageFolder(train_dir, transform=transform)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

# Model Definition

print("Defining LULC_Model class...")

class LULC_Model(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        #self.network = models.wide_resnet50_2(pretrained=pretrained)
        self.network = models.wide_resnet50_2(weights=Wide_ResNet50_2_Weights.IMAGENET1K_V1)
        n_inputs = self.network.fc.in_features
        self.network.fc = nn.Sequential(
            nn.Linear(n_inputs, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, NUM_CLASSES)
        )

    def forward(self, xb):
        return self.network(xb)

    def freeze(self):
        for param in self.network.parameters():
            param.requires_grad = False
        for param in self.network.fc.parameters():
            param.requires_grad = True

def get_model():
    use_cuda = torch.cuda.is_available()
    DEVICE = torch.device('cuda' if use_cuda else 'cpu')
    model = LULC_Model()
    model.freeze()  # Freeze the entire model except the FC layer
    return model.to(DEVICE)

def get_model(load_weights=False):
    use_cuda = torch.cuda.is_available()
    DEVICE = torch.device('cuda' if use_cuda else 'cpu')
    model = LULC_Model()
    if load_weights:
        model.load_state_dict(torch.load(config.MODEL_PATH))
        model.eval()
    
    return model.to(DEVICE)


# Training and Evaluation

print("Defining helper functions for training and evaluation...")

def train_model(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        correct += (preds == labels).sum().item()
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = correct / len(dataloader.dataset)
    return epoch_loss, epoch_acc

def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            correct += (preds == labels).sum().item()
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = correct / len(dataloader.dataset)
    return epoch_loss, epoch_acc

# Main Training Loop

print("Beginning Main Training Loop...")

if __name__ == "__main__":
    print("Setting up device...")
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {DEVICE}")

    print("Initializing model...")
    model = get_model()

    print("Setting up loss function and optimizer...")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 10

    for epoch in range(num_epochs):
        print(f"Starting Epoch {epoch+1}/{num_epochs}...")
        print("Training...")
        train_loss, train_acc = train_model(model, train_loader, criterion, optimizer, DEVICE)
        print("Evaluating...")
        val_loss, val_acc = evaluate_model(model, val_loader, criterion, DEVICE)
        print(f"Epoch {epoch+1}/{num_epochs} - "
              f"Train Loss: {train_loss:.4f} - Train Acc: {train_acc:.4f} - "
              f"Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f}")

    print("Training complete. Saving the trained model...")
    torch.save(model.state_dict(), r"C:\Users\hryad\Desktop\iitm\Space Apps\Land-Cover-Classification-using-Sentinel-2-Dataset\data\model")
    print("Model saved successfully!")