import os
import json
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
import timm
from tqdm import tqdm
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

# 1. Hyperparameters & Configuration
DATA_DIR = "./dataset"  
BATCH_SIZE = 16         
NUM_EPOCHS = 30         
LEARNING_RATE = 5e-4
IMAGE_SIZE = 224
SAVE_PATH = "skin_disease_model_final.pth"
CLASSES_PATH = "classes.json"

# Hardware Setup
DEVICE = torch.device("cpu")
print(f"Targeting execution device: {DEVICE}")


# 2. Advanced Data Augmentations
# Prevents model from memorizing specific background textures or colors
train_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(degrees=45),
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
    transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.05),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 3. Load Data & Address Imbalance
def get_dataloaders():
    train_dataset = ImageFolder(root=os.path.join(DATA_DIR, 'train'), transform=train_transforms)
    val_dataset = ImageFolder(root=os.path.join(DATA_DIR, 'val'), transform=val_transforms)

    # Exporting class names to JSON 
    class_names = train_dataset.classes
    with open(CLASSES_PATH, "w") as f:
        json.dump(class_names, f, indent=4)
    print(f"--> Saved class mapping to {CLASSES_PATH}")
    print(f"Classes ({len(class_names)}): {class_names}")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=False)

    # Compute balanced weights for class imbalance
    targets = train_dataset.targets
    class_weights = compute_class_weight('balanced', classes=np.unique(targets), y=targets)
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(DEVICE)

    return train_loader, val_loader, class_weights, len(class_names)

# 4. Model Architecture (Lighter for CPU)
def build_model(num_classes):
    print("Loading pre-trained EfficientNet-B0...")
    model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=num_classes)
    return model

# 5. Training Loop with Anti-Overconfidence
def train_model():
    train_loader, val_loader, class_weights, num_classes = get_dataloaders()
    model = build_model(num_classes).to(DEVICE)

    # label_smoothing=0.1 directly penalizes 99%+ overconfidence outputs
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)

    best_val_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())

    print(f"\nStarting training on {DEVICE} for {NUM_EPOCHS} epochs...\n" + "="*45)

    for epoch in range(NUM_EPOCHS):
        # --- Train Phase ---
        model.train()
        running_loss = 0.0
        correct_preds = 0
        total_preds = 0

        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1:02d}/{NUM_EPOCHS:02d} [Train]")
        for inputs, labels in train_bar:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()

            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            correct_preds += torch.sum(preds == labels.data)
            total_preds += inputs.size(0)

            train_bar.set_postfix(loss=loss.item())

        scheduler.step()
        epoch_train_loss = running_loss / total_preds
        epoch_train_acc = (correct_preds.double() / total_preds).item()

        # --- Validation Phase ---
        model.eval()
        val_loss = 0.0
        val_corrects = 0
        val_total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                _, preds = torch.max(outputs, 1)
                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data)
                val_total += inputs.size(0)

        epoch_val_loss = val_loss / val_total
        epoch_val_acc = (val_corrects.double() / val_total).item()

        print(f"Epoch {epoch+1:02d}/{NUM_EPOCHS:02d} | "
              f"Train Loss: {epoch_train_loss:.4f} Acc: {epoch_train_acc:.4f} | "
              f"Val Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc:.4f}")

        # Save best model checkpoint
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"  --> [Saved] New peak validation accuracy: {best_val_acc:.4f}")

    print("\nTraining Complete!")
    print(f"Best Validation Accuracy Reached: {best_val_acc:.4f}")

    return model

if __name__ == "__main__":
    train_model()