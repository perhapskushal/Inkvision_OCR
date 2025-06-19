import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import numpy as np
from datetime import datetime
from typing import Tuple, List, Dict
import logging
from tqdm import tqdm
import torch.cuda.amp as amp

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('script_classifier.log'),
        logging.StreamHandler()
    ]
)

def setup_gpu():
    """Setup GPU and return device."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Please check your GPU installation.")
    
    # Set CUDA device
    torch.cuda.set_device(0)  # Use first GPU
    device = torch.device('cuda')
    
    # Print GPU information
    logging.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    logging.info(f"CUDA Version: {torch.version.cuda}")
    logging.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Enable TF32 for better performance on Ampere GPUs
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    return device

def find_images_in_dir(directory: str) -> List[str]:
    """Recursively find all image files in a directory and its subdirectories."""
    image_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_files.append(os.path.join(root, file))
    return image_files

class ScriptDataset(Dataset):
    def __init__(self, root_dirs: List[str], transform=None):
        """
        Args:
            root_dirs: List of root directories containing the images
            transform: Optional transform to be applied on images
        """
        self.transform = transform
        self.image_paths = []
        self.labels = []
        
        # Map folder names to labels
        self.label_map = {
            'IIIT5k': 0,  # English
            'archive': 1  # Devanagari
        }
        
        # Collect all images and their labels
        for root_dir in root_dirs:
            folder_name = os.path.basename(root_dir)
            if folder_name not in self.label_map:
                logging.warning(f"Skipping unknown directory: {folder_name}")
                continue
            
            label = self.label_map[folder_name]
            
            # Handle IIIT5k directory structure
            if folder_name == 'IIIT5k':
                test_dir = os.path.join(root_dir, 'test')
                if os.path.exists(test_dir):
                    for img_name in os.listdir(test_dir):
                        if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                            self.image_paths.append(os.path.join(test_dir, img_name))
                            self.labels.append(label)
            
            # Handle archive directory structure
            elif folder_name == 'archive':
                images_dir = os.path.join(root_dir, 'Images', 'Images')
                if os.path.exists(images_dir):
                    # Recursively find all images in subdirectories
                    image_files = find_images_in_dir(images_dir)
                    self.image_paths.extend(image_files)
                    self.labels.extend([label] * len(image_files))
        
        if len(self.image_paths) == 0:
            raise ValueError("No valid images found in the specified directories. Please check the directory structure and image files.")
        
        logging.info(f"Loaded {len(self.image_paths)} images:")
        logging.info(f"English (IIIT5k): {sum(1 for label in self.labels if label == 0)}")
        logging.info(f"Devanagari (archive): {sum(1 for label in self.labels if label == 1)}")
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        try:
            # Load and transform image
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, label
        except Exception as e:
            logging.error(f"Error loading image {img_path}: {str(e)}")
            raise

class ScriptClassifier(nn.Module):
    def __init__(self, num_classes: int = 2):
        super(ScriptClassifier, self).__init__()
        # Use ResNet-18 as the base model
        self.model = models.resnet18(pretrained=True)
        
        # Modify the final layer for binary classification
        num_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

def calculate_class_weights(labels: List[int]) -> torch.Tensor:
    """Calculate class weights to handle imbalance."""
    class_counts = torch.bincount(torch.tensor(labels))
    total_samples = len(labels)
    class_weights = total_samples / (len(class_counts) * class_counts)
    return class_weights

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    num_epochs: int,
    device: torch.device,
    save_dir: str
) -> Dict[str, List[float]]:
    """Train the model and save checkpoints."""
    best_val_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    # Initialize mixed precision training
    scaler = amp.GradScaler()
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        class_correct = [0, 0]  # For English and Devanagari
        class_total = [0, 0]    # For English and Devanagari
        
        # Create progress bar for training
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            # Use mixed precision training
            with amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            # Scale loss and backpropagate
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            # Track per-class accuracy
            for i in range(2):  # 2 classes
                mask = (labels == i)
                class_correct[i] += (predicted[mask] == labels[mask]).sum().item()
                class_total[i] += mask.sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100 * train_correct / train_total:.2f}%'
            })
        
        train_loss = train_loss / len(train_loader)
        train_acc = 100 * train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_class_correct = [0, 0]  # For English and Devanagari
        val_class_total = [0, 0]    # For English and Devanagari
        
        # Create progress bar for validation
        pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]')
        
        with torch.no_grad():
            for images, labels in pbar:
                images, labels = images.to(device), labels.to(device)
                
                # Use mixed precision for validation
                with amp.autocast():
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
                # Track per-class accuracy
                for i in range(2):  # 2 classes
                    mask = (labels == i)
                    val_class_correct[i] += (predicted[mask] == labels[mask]).sum().item()
                    val_class_total[i] += mask.sum().item()
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100 * val_correct / val_total:.2f}%'
                })
        
        val_loss = val_loss / len(val_loader)
        val_acc = 100 * val_correct / val_total
        
        # Update history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Log detailed metrics
        logging.info(f'Epoch [{epoch+1}/{num_epochs}]')
        logging.info(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        logging.info(f'Train Class Acc - English: {100 * class_correct[0] / class_total[0]:.2f}%, Devanagari: {100 * class_correct[1] / class_total[1]:.2f}%')
        logging.info(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        logging.info(f'Val Class Acc - English: {100 * val_class_correct[0] / val_class_total[0]:.2f}%, Devanagari: {100 * val_class_correct[1] / val_class_total[1]:.2f}%')
        
        # Save model if validation accuracy improves
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            save_path = os.path.join(save_dir, f'model_{timestamp}.pth')
            
            # Create save directory if it doesn't exist
            os.makedirs(save_dir, exist_ok=True)
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'scaler': scaler.state_dict(),  # Save mixed precision scaler state
            }, save_path)
            logging.info(f'Model saved to {save_path}')
    
    return history

def predict_script(image_path: str, model_path: str) -> str:
    """Returns either 'devanagari' or 'english'"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load and prepare the model
    model = ScriptClassifier()
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Prepare image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)
    
    # Make prediction
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        
    return 'devanagari' if predicted.item() == 1 else 'english'

def main():
    parser = argparse.ArgumentParser(description='Train script classifier')
    parser.add_argument('--epochs', type=int, default=10,
                      help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=128,  # Increased batch size for RTX 4060
                      help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.001,
                      help='Learning rate')
    parser.add_argument('--image_size', type=int, default=224,
                      help='Size to resize images to')
    parser.add_argument('--save_dir', type=str, default='models',
                      help='Directory to save model checkpoints')
    parser.add_argument('--num_workers', type=int, default=4,
                      help='Number of workers for data loading')
    args = parser.parse_args()
    
    # Setup GPU
    try:
        device = setup_gpu()
    except RuntimeError as e:
        logging.error(str(e))
        return
    
    # Define transforms with data augmentation for training
    train_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Simpler transforms for validation
    val_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    root_dirs = ['IIIT5k', 'archive']
    dataset = ScriptDataset(root_dirs, transform=train_transform)
    
    # Calculate class weights
    class_weights = calculate_class_weights(dataset.labels)
    logging.info(f"Class weights: English={class_weights[0]:.2f}, Devanagari={class_weights[1]:.2f}")
    
    # Split dataset into train and validation
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Update validation dataset transform
    val_dataset.dataset.transform = val_transform
    
    # Create data loaders with pin_memory for faster GPU transfer
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size,
        shuffle=True, 
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True  # Keep workers alive between epochs
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size,
        shuffle=False, 
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True
    )
    
    # Initialize model, criterion, and optimizer
    model = ScriptClassifier().to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    
    # Use AdamW optimizer with weight decay
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=0.01,
        betas=(0.9, 0.999)
    )
    
    # Add learning rate scheduler
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        epochs=args.epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.3,
        anneal_strategy='cos'
    )
    
    # Train the model
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=args.epochs,
        device=device,
        save_dir=args.save_dir
    )

if __name__ == '__main__':
    main() 