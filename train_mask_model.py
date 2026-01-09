# ==================== TRAIN_MASK_MODEL.PY ====================
"""
Complete Training Script for Mask Detection CNN Model (PyTorch)
Trains a 5-class mask detection classifier from scratch

Usage:
    python train_mask_model.py --data_dir data/Dataset/train --epochs 50 --batch_size 32
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

class MaskDetectionCNN(nn.Module):
    """CNN model for mask detection"""
    def __init__(self, num_classes=5):
        super(MaskDetectionCNN, self).__init__()
        
        # Block 1
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25)
        )
        
        # Block 2
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25)
        )
        
        # Block 3
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25)
        )
        
        # Block 4
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25)
        )
        
        # Dense layers
        self.fc = nn.Sequential(
            nn.Linear(128 * 8 * 8, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        return x

class MaskDetectionModelTrainer:
    def __init__(self, input_size=128, num_classes=5):
        self.input_size = input_size
        self.num_classes = num_classes
        
        # Detect best available device
        if torch.backends.mps.is_available():
            self.device = torch.device('mps')  # Apple Silicon GPU
        elif torch.cuda.is_available():
            self.device = torch.device('cuda')  # NVIDIA GPU
        else:
            self.device = torch.device('cpu')
        
        print(f"Using device: {self.device}")
        
        self.model = None
        self.history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        
        # Class names mapping for the dataset (alphabetical order)
        # Dataset folders: cloth, n95, n95v, nfm, srg
        self.class_names = ['Cloth Mask', 'N95 Mask', 'Partial Mask', 'No Mask', 'Surgical Mask']
    
    def create_model(self):
        """Create CNN model for mask detection"""
        print("\nCreating CNN model...")
        self.model = MaskDetectionCNN(num_classes=self.num_classes).to(self.device)
        
        # Print model summary
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}\n")
        
        return self.model
    
    def create_data_loaders(self, train_dir, batch_size=32, validation_split=0.2):
        """Create data loaders with augmentation"""
        print(f"Creating data loaders from {train_dir}...")
        
        # Training data augmentation
        train_transform = transforms.Compose([
            transforms.Resize((self.input_size, self.input_size)),
            transforms.RandomRotation(30),
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(degrees=0, translate=(0.3, 0.3), shear=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Validation data (only resize and normalize)
        val_transform = transforms.Compose([
            transforms.Resize((self.input_size, self.input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Load full dataset with train transforms initially
        full_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
        
        # Split into train and validation
        dataset_size = len(full_dataset)
        val_size = int(validation_split * dataset_size)
        train_size = dataset_size - val_size
        
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
        
        # Update validation dataset transform
        val_dataset.dataset = datasets.ImageFolder(train_dir, transform=val_transform)
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=0,  # Set to 0 for Mac compatibility
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False,
            num_workers=0,  # Set to 0 for Mac compatibility
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        print(f"Training samples: {train_size}")
        print(f"Validation samples: {val_size}")
        print(f"Classes: {full_dataset.classes}")
        print(f"Batch size: {batch_size}\n")
        
        return train_loader, val_loader
    
    def train_epoch(self, train_loader, criterion, optimizer):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc='Training', ncols=100)
        for inputs, labels in pbar:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}', 
                'acc': f'{100 * correct / total:.2f}%'
            })
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        return epoch_loss, epoch_acc
    
    def validate(self, val_loader, criterion):
        """Validate the model"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc='Validation', ncols=100):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / len(val_loader)
        epoch_acc = 100 * correct / total
        return epoch_loss, epoch_acc
    
    def train(self, train_dir, epochs=50, batch_size=32, validation_split=0.2, 
              learning_rate=1e-3, patience=10):
        """Train the model"""
        if self.model is None:
            self.create_model()
        
        # Create data loaders
        train_loader, val_loader = self.create_data_loaders(
            train_dir, batch_size, validation_split
        )
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        
        # Training loop
        best_val_acc = 0.0
        patience_counter = 0
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        os.makedirs('models', exist_ok=True)
        best_model_path = f'models/mask_detection_best_{timestamp}.pth'
        
        print("Starting training...")
        print("=" * 60)
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            print("-" * 60)
            
            # Train
            train_loss, train_acc = self.train_epoch(train_loader, criterion, optimizer)
            
            # Validate
            val_loss, val_acc = self.validate(val_loader, criterion)
            
            # Update learning rate
            scheduler.step(val_loss)
            
            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            print(f"\nTrain Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_acc,
                    'class_names': self.class_names
                }, best_model_path)
                print(f"✅ Saved best model with val_acc: {val_acc:.2f}%")
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= patience:
                print(f"\n⚠️  Early stopping triggered after {epoch+1} epochs")
                break
        
        print("\n" + "=" * 60)
        print(f"Training completed! Best validation accuracy: {best_val_acc:.2f}%")
        
        return self.history
    
    def save_model(self, save_path='models/mask_detection_model.pth'):
        """Save trained model"""
        if self.model is None:
            print("No model to save. Train first!")
            return
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'class_names': self.class_names,
            'input_size': self.input_size
        }, save_path)
        print(f"Model saved to {save_path}")
    
    def plot_training_history(self, save_path='training_history.png'):
        """Plot training history"""
        if not self.history['train_loss']:
            print("No training history available")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Loss
        axes[0].plot(self.history['train_loss'], label='Train Loss', marker='o')
        axes[0].plot(self.history['val_loss'], label='Validation Loss', marker='s')
        axes[0].set_title('Model Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Accuracy
        axes[1].plot(self.history['train_acc'], label='Train Accuracy', marker='o')
        axes[1].plot(self.history['val_acc'], label='Validation Accuracy', marker='s')
        axes[1].set_title('Model Accuracy')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy (%)')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nTraining history plot saved to {save_path}")
        plt.close()

def main():
    parser = argparse.ArgumentParser(description='Train mask detection model (PyTorch)')
    parser.add_argument('--data_dir', type=str, default='data/Dataset/train',
                       help='Path to training data directory')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--validation_split', type=float, default=0.2,
                       help='Validation split ratio')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--save_path', type=str, default='models/mask_detection_model.pth',
                       help='Path to save model')
    
    args = parser.parse_args()
    
    # Check if data directory exists
    if not os.path.exists(args.data_dir):
        print(f"❌ Error: Data directory {args.data_dir} not found!")
        print("\nExpected directory structure:")
        print(f"{args.data_dir}/")
        print("  ├── cloth/      (Cloth Mask)")
        print("  ├── n95/        (N95 Mask)")
        print("  ├── n95v/       (Partial Mask)")
        print("  ├── nfm/        (No Mask)")
        print("  └── srg/        (Surgical Mask)")
        sys.exit(1)
    
    print("=" * 60)
    print("MASK DETECTION MODEL TRAINING (PyTorch)")
    print("=" * 60)
    
    # Create trainer
    trainer = MaskDetectionModelTrainer()
    
    # Create and train model
    trainer.create_model()
    trainer.train(
        train_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_split=args.validation_split,
        learning_rate=args.learning_rate
    )
    
    # Save model
    trainer.save_model(args.save_path)
    
    # Plot results
    trainer.plot_training_history()
    
    print("\n" + "=" * 60)
    print("✅ TRAINING COMPLETE!")
    print("=" * 60)
    print(f"Model saved to: {args.save_path}")
    print(f"Training plot saved to: training_history.png")

if __name__ == '__main__':
    main()
