import os
import shutil
import pandas as pd
import logging
from pathlib import Path
from sklearn.model_selection import train_test_split
import subprocess
from tqdm import tqdm    
import time
import cv2
import multiprocessing
from functools import partial
import tempfile
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import warnings
from PIL import Image

# Suppress specific UserWarning related to deprecated pytree node registration
warnings.filterwarnings("ignore", category=UserWarning, message=".*_register_pytree_node is deprecated.*")

# Suppress the specific deprecation warning
warnings.filterwarnings("ignore", message="torch.utils._pytree._register_pytree_node is deprecated")
warnings.filterwarnings("ignore", message="torch.utils._pytree._register_pytree_node is deprecated. Please use torch.utils._pytree.register_pytree_node instead.")

# Enable Windows long path support
if os.name == 'nt':
    import ctypes
    from ctypes import wintypes
    kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel32.SetFileAttributesW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD]
    kernel32.SetFileAttributesW.restype = wintypes.BOOL
    # Enable long paths
    kernel32.SetFileAttributesW(str(Path.cwd()), 0x00000800)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('tesseract_finetune.log'),
        logging.StreamHandler()
    ]
)

class NeuralNetFinetuner:
    def __init__(self, data_dir='archive/Images/Images', output_dir='out', batch_size=64, epochs=50, lr=1e-4, max_images=None):
        # Set specific GPU (RTX 4060)
        if torch.cuda.is_available():
            torch.cuda.set_device(0)  # Use first GPU (RTX 4060)
            logging.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.max_images = max_images
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.output_dir.mkdir(exist_ok=True)

    def build_model(self, num_classes):
        # Use ResNet18 with better regularization
        model = models.resnet18(weights=None)
        # Modify first layer for grayscale
        model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # Add dropout and batch norm before final layer
        model.fc = nn.Sequential(
            nn.BatchNorm1d(model.fc.in_features),
            nn.Dropout(0.5),
            nn.Linear(model.fc.in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        return model.to(self.device)

    def prepare_dataloaders(self):
        # Add data augmentation for training
        train_transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((32, 32)),
            transforms.RandomRotation(10),
            transforms.RandomAffine(0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        
        val_transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        
        # Create datasets with transforms
        train_dataset = datasets.ImageFolder(str(self.data_dir), transform=train_transform)
        val_dataset = datasets.ImageFolder(str(self.data_dir), transform=val_transform)
        test_dataset = datasets.ImageFolder(str(self.data_dir), transform=val_transform)
        
        if self.max_images is not None:
            # Calculate split sizes
            total_size = min(len(train_dataset), self.max_images)
            train_size = int(0.7 * total_size)
            val_size = int(0.15 * total_size)
            test_size = total_size - train_size - val_size
            
            # Create indices for splitting
            indices = torch.randperm(total_size)
            train_indices = indices[:train_size]
            val_indices = indices[train_size:train_size + val_size]
            test_indices = indices[train_size + val_size:]
            
            # Create subsets
            train_dataset = torch.utils.data.Subset(train_dataset, train_indices)
            val_dataset = torch.utils.data.Subset(val_dataset, val_indices)
            test_dataset = torch.utils.data.Subset(test_dataset, test_indices)
        else:
            # Calculate split sizes
            total_size = len(train_dataset)
            train_size = int(0.7 * total_size)
            val_size = int(0.15 * total_size)
            test_size = total_size - train_size - val_size
            
            # Create indices for splitting
            indices = torch.randperm(total_size)
            train_indices = indices[:train_size]
            val_indices = indices[train_size:train_size + val_size]
            test_indices = indices[train_size + val_size:]
            
            # Create subsets
            train_dataset = torch.utils.data.Subset(train_dataset, train_indices)
            val_dataset = torch.utils.data.Subset(val_dataset, val_indices)
            test_dataset = torch.utils.data.Subset(test_dataset, test_indices)
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=2)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=2)
        
        self.class_names = train_dataset.dataset.classes
        return train_loader, val_loader, test_loader

    def train(self):
        logging.info("Starting training process...")
        logging.info("Preparing dataloaders...")
        train_loader, val_loader, test_loader = self.prepare_dataloaders()
        logging.info("Dataloaders prepared.")
        logging.info(f"Train dataset size: {len(train_loader.dataset)}")
        logging.info(f"Validation dataset size: {len(val_loader.dataset)}")
        logging.info(f"Test dataset size: {len(test_loader.dataset)}")

        num_classes = len(self.class_names)
        logging.info(f"Number of classes: {num_classes}")
        model = self.build_model(num_classes)
        logging.info("Model built.")
        criterion = nn.CrossEntropyLoss()
        # Use AdamW optimizer with weight decay
        optimizer = optim.AdamW(model.parameters(), lr=self.lr, weight_decay=0.01)
        # Use cosine annealing learning rate scheduler with warm restarts
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, 
            T_0=5,  # Restart every 5 epochs
            T_mult=2,  # Double the restart interval after each restart
            eta_min=1e-6  # Minimum learning rate
        )
        best_val_acc = 0.0
        best_model_path = self.output_dir / 'nep_finetuned.pth'
        
        # Early stopping parameters
        patience = 10
        patience_counter = 0
        
        logging.info("Starting epoch loop.")
        for epoch in range(self.epochs):
            logging.info(f"--- Epoch {epoch+1}/{self.epochs} ---")
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            for i, (images, labels) in enumerate(train_loader):
                if i == 0:
                    logging.info("Processing first batch of training data...")
                    logging.info(f"Images batch shape: {images.shape}, Labels batch shape: {labels.shape}")
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)
            train_loss = running_loss / total
            train_acc = correct / total
            
            # Validation
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item() * images.size(0)
                    _, predicted = outputs.max(1)
                    val_correct += predicted.eq(labels).sum().item()
                    val_total += labels.size(0)
            val_loss /= val_total
            val_acc = val_correct / val_total
            
            # Update learning rate
            scheduler.step()
            
            logging.info(f"Epoch {epoch+1}/{self.epochs} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), best_model_path)
                logging.info(f"Best model saved to {best_model_path}")
                patience_counter = 0
            else:
                patience_counter += 1
                
            # Early stopping
            if patience_counter >= patience:
                logging.info(f"Early stopping triggered after {epoch + 1} epochs")
                break
        
        # Load best model for test
        model.load_state_dict(torch.load(best_model_path, map_location=self.device))
        self.evaluate(model, test_loader)

    def evaluate(self, model, test_loader):
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = model(images)
                _, predicted = outputs.max(1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        acc = (np.array(all_preds) == np.array(all_labels)).mean()
        logging.info(f"Test Accuracy: {acc:.4f}")

    def predict(self, image_path):
        """Predict the class of a single image."""
        # Load and prepare the model
        model = self.build_model(len(self.class_names))
        model.load_state_dict(torch.load(self.output_dir / 'nep_finetuned.pth', map_location=self.device))
        model.eval()

        # Prepare the image
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        
        # Load and transform the image
        image = Image.open(image_path).convert('L')
        image_tensor = transform(image).unsqueeze(0).to(self.device)
        
        # Make prediction
        with torch.no_grad():
            output = model(image_tensor)
            _, predicted = output.max(1)
            predicted_class = self.class_names[predicted.item()]
            
        return predicted_class

def main():
    import argparse
    parser = argparse.ArgumentParser(description='PyTorch Neural Network Finetuning')
    parser.add_argument('--max_images', type=int, default=None, help='Max images to use (for quick test). Use all images if None.')
    parser.add_argument('--epochs', type=int, default=50, help='Epochs for PyTorch training')
    parser.add_argument('--predict', type=str, help='Path to image for prediction')
    args = parser.parse_args()
    
    if args.predict:
        # Initialize the model and make prediction
        nn_finetuner = NeuralNetFinetuner()
        prediction = nn_finetuner.predict(args.predict)
        logging.info(f"Predicted class: {prediction}")
    else:
        nn_finetuner = NeuralNetFinetuner(max_images=args.max_images, epochs=args.epochs)
        nn_finetuner.train()

if __name__ == "__main__":
    main() 