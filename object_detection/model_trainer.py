"""
Neural network model training module.
Provides utilities for training custom object detection and classification models.
"""
import os
import sys
import time
import json
import random
import numpy as np
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import cv2

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision
from torchvision import transforms, models
from torchvision.datasets import ImageFolder

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class CustomImageDataset(Dataset):
    """
    Custom dataset for loading images for model training.
    Supports both folder-based (ImageFolder style) and JSON annotation formats.
    """
    
    def __init__(self, data_path, transform=None, annotation_file=None):
        """
        Initialize the dataset.
        
        Args:
            data_path: Path to the data directory
            transform: Pytorch transforms to apply to images
            annotation_file: Optional path to JSON annotation file
        """
        self.data_path = Path(data_path)
        self.transform = transform
        self.samples = []
        self.classes = []
        self.class_to_idx = {}
        
        # Check if using annotations or folder structure
        if annotation_file:
            self._load_from_annotations(annotation_file)
        else:
            self._load_from_folders()
    
    def _load_from_folders(self):
        """Load dataset from folder structure (each class in its own folder)."""
        # Get list of class folders
        class_folders = [f for f in self.data_path.iterdir() if f.is_dir()]
        self.classes = [folder.name for folder in class_folders]
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        # Collect all image paths and their labels
        for class_folder in class_folders:
            class_idx = self.class_to_idx[class_folder.name]
            
            # Get all images in this class folder
            for img_path in class_folder.glob("*.*"):
                if self._is_valid_image(img_path):
                    self.samples.append((str(img_path), class_idx))
    
    def _load_from_annotations(self, annotation_file):
        """Load dataset from a JSON annotation file."""
        with open(annotation_file, 'r') as f:
            annotations = json.load(f)
        
        # Extract unique classes
        unique_classes = set()
        for item in annotations:
            unique_classes.add(item['category'])
        
        self.classes = sorted(list(unique_classes))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        # Process annotations
        for item in annotations:
            img_path = os.path.join(self.data_path, item['image_path'])
            if os.path.exists(img_path) and self._is_valid_image(img_path):
                class_idx = self.class_to_idx[item['category']]
                self.samples.append((img_path, class_idx))
    
    def _is_valid_image(self, path):
        """Check if file is a valid image."""
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
        return Path(path).suffix.lower() in valid_extensions
    
    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        Get a sample from the dataset.
        
        Args:
            idx: Sample index
            
        Returns:
            tuple: (image, label)
        """
        img_path, label = self.samples[idx]
        
        # Load image
        image = cv2.imread(img_path)
        # Convert from BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply transforms if any
        if self.transform:
            image = self.transform(image)
        
        return image, label


def get_transforms(train=True, input_size=None):
    """
    Get image transforms for training or validation.
    
    Args:
        train: Whether to get transforms for training (with augmentation) or validation
        input_size: Input size for the model, defaults to config value
        
    Returns:
        PyTorch transforms composition
    """
    if input_size is None:
        input_size = config.MODEL_INPUT_SIZE
    
    # Normalization parameters for pre-trained models
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    
    # Basic transforms for both training and validation
    basic_transforms = [
        transforms.ToPILImage(),
        transforms.Resize(input_size),
        transforms.ToTensor(),
        normalize
    ]
    
    # Add augmentation for training if enabled
    if train and config.DATA_AUGMENTATION:
        train_transforms = [
            transforms.ToPILImage(),
            transforms.RandomResizedCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomRotation(20),
            transforms.ToTensor(),
            normalize
        ]
        return transforms.Compose(train_transforms)
    
    return transforms.Compose(basic_transforms)


def create_model(num_classes, architecture=None, pretrained=True):
    """
    Create a model for training.
    
    Args:
        num_classes: Number of output classes
        architecture: Architecture to use, defaults to config value
        pretrained: Whether to use pretrained weights
        
    Returns:
        PyTorch model
    """
    if architecture is None:
        architecture = config.MODEL_ARCHITECTURE
    
    # Create model based on specified architecture
    if architecture == "mobilenet_v2":
        model = models.mobilenet_v2(pretrained=pretrained)
        # Modify the classifier for the number of classes
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    
    elif architecture == "resnet18":
        model = models.resnet18(pretrained=pretrained)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    elif architecture == "resnet50":
        model = models.resnet50(pretrained=pretrained)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        
    elif architecture == "efficientnet_b0":
        model = models.efficientnet_b0(pretrained=pretrained)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    
    else:
        # Default to MobileNetV2
        print(f"Architecture {architecture} not recognized, using MobileNetV2")
        model = models.mobilenet_v2(pretrained=pretrained)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    
    return model


class ModelTrainer:
    """Main class for training neural network models."""
    
    def __init__(self, data_path=None, batch_size=None, learning_rate=None, 
                num_epochs=None, architecture=None, device=None):
        """
        Initialize the model trainer.
        
        Args:
            data_path: Path to the training data
            batch_size: Batch size for training
            learning_rate: Learning rate for optimization
            num_epochs: Number of training epochs
            architecture: Model architecture to use
            device: Device to use for training ('cuda' or 'cpu')
        """
        # Use defaults from config if not provided
        self.data_path = data_path or config.TRAINING_DATA_PATH
        self.batch_size = batch_size or config.BATCH_SIZE
        self.learning_rate = learning_rate or config.LEARNING_RATE
        self.num_epochs = num_epochs or config.NUM_EPOCHS
        self.architecture = architecture or config.MODEL_ARCHITECTURE
        
        # Set device for training
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        
        # Initialize datasets, model, optimizer
        self.train_dataset = None
        self.val_dataset = None
        self.train_loader = None
        self.val_loader = None
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        
        # Create output directory for checkpoints
        self.output_dir = os.path.dirname(config.MODEL_PATH)
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }
    
    def prepare_data(self, annotation_file=None, val_split=None):
        """
        Prepare datasets and dataloaders for training.
        
        Args:
            annotation_file: Optional path to annotation file
            val_split: Validation set proportion, defaults to config value
        """
        if val_split is None:
            val_split = config.VALIDATION_SPLIT
        
        # Get transforms for training and validation
        train_transform = get_transforms(train=True)
        val_transform = get_transforms(train=False)
        
        # Create full dataset
        full_dataset = CustomImageDataset(
            self.data_path, 
            transform=train_transform,
            annotation_file=annotation_file
        )
        
        # Set class names
        self.classes = full_dataset.classes
        self.num_classes = len(self.classes)
        
        print(f"Dataset loaded with {len(full_dataset)} images in {self.num_classes} classes:")
        for i, class_name in enumerate(self.classes):
            count = sum(1 for _, label in full_dataset.samples if label == i)
            print(f"  - {class_name}: {count} images")
        
        # Split into training and validation sets
        val_size = int(val_split * len(full_dataset))
        train_size = len(full_dataset) - val_size
        
        # Use random_split to create the split
        self.train_dataset, self.val_dataset = random_split(
            full_dataset, 
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)  # For reproducibility
        )
        
        # Apply different transforms to validation set
        self.val_dataset.dataset = CustomImageDataset(
            self.data_path,
            transform=val_transform,
            annotation_file=annotation_file
        )
        # Make sure validation set uses same samples as the split
        val_indices = self.val_dataset.indices
        self.val_dataset.dataset.samples = [self.val_dataset.dataset.samples[i] for i in val_indices]
        
        print(f"Training set: {len(self.train_dataset)} images")
        print(f"Validation set: {len(self.val_dataset)} images")
        
        # Create data loaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True if self.device.type == 'cuda' else False
        )
    
    def setup_model(self):
        """Setup the model, loss function, optimizer and scheduler."""
        # Create model
        self.model = create_model(
            num_classes=self.num_classes,
            architecture=self.architecture,
            pretrained=True
        )
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Define loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Define optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=1e-4
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.1,
            patience=5,
            verbose=True
        )
    
    def train_epoch(self):
        """
        Train for one epoch.
        
        Returns:
            avg_loss, accuracy: Average loss and accuracy for the epoch
        """
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Progress tracking
        start_time = time.time()
        
        for i, (inputs, labels) in enumerate(self.train_loader):
            # Move data to device
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            self.optimizer.step()
            
            # Update statistics
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Report progress
            if (i + 1) % 10 == 0:
                elapsed = time.time() - start_time
                print(f"Batch {i+1}/{len(self.train_loader)}, "
                     f"Loss: {loss.item():.4f}, "
                     f"Accuracy: {100 * correct / total:.2f}%, "
                     f"Time: {elapsed:.2f}s")
        
        # Calculate average loss and accuracy
        epoch_loss = running_loss / total
        epoch_acc = 100 * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate(self):
        """
        Validate the model.
        
        Returns:
            avg_loss, accuracy: Average loss and accuracy for validation set
        """
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in self.val_loader:
                # Move data to device
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                
                # Update statistics
                running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        # Calculate average loss and accuracy
        val_loss = running_loss / total
        val_acc = 100 * correct / total
        
        return val_loss, val_acc
    
    def train(self):
        """
        Train the model for the specified number of epochs.
        
        Returns:
            history: Training history
        """
        # Setup model if not already done
        if self.model is None:
            self.setup_model()
        
        print(f"Starting training for {self.num_epochs} epochs...")
        best_val_loss = float('inf')
        
        for epoch in range(self.num_epochs):
            print(f"\nEpoch {epoch+1}/{self.num_epochs}")
            print("-" * 20)
            
            # Train for one epoch
            train_loss, train_acc = self.train_epoch()
            
            # Validate
            val_loss, val_acc = self.validate()
            
            # Update learning rate based on validation loss
            self.scheduler.step(val_loss)
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
            # Save checkpoint if this is the best model so far
            if val_loss < best_val_loss and config.SAVE_CHECKPOINTS:
                best_val_loss = val_loss
                self.save_checkpoint(f"best_model_epoch_{epoch+1}.pth")
                print(f"Saved checkpoint at epoch {epoch+1}")
        
        print("Training complete!")
        
        # Save final model
        self.save_model()
        
        # Save and plot training history
        self.save_history()
        self.plot_history()
        
        return self.history
    
    def save_checkpoint(self, filename=None):
        """
        Save a training checkpoint.
        
        Args:
            filename: Checkpoint filename
        """
        if filename is None:
            filename = f"checkpoint_epoch_{len(self.history['train_loss'])}.pth"
        
        checkpoint_path = os.path.join(self.output_dir, filename)
        
        # Create checkpoint
        checkpoint = {
            'epoch': len(self.history['train_loss']),
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_loss': self.history['train_loss'],
            'val_loss': self.history['val_loss'],
            'train_acc': self.history['train_acc'],
            'val_acc': self.history['val_acc'],
            'classes': self.classes
        }
        
        # Save checkpoint
        torch.save(checkpoint, checkpoint_path)
    
    def load_checkpoint(self, checkpoint_path):
        """
        Load a training checkpoint.
        
        Args:
            checkpoint_path: Path to the checkpoint file
        """
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Setup model if not already done
        if self.model is None:
            self.classes = checkpoint['classes']
            self.num_classes = len(self.classes)
            self.setup_model()
        
        # Load model state
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state if resuming training
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Load history
        self.history['train_loss'] = checkpoint['train_loss']
        self.history['val_loss'] = checkpoint['val_loss']
        self.history['train_acc'] = checkpoint['train_acc']
        self.history['val_acc'] = checkpoint['val_acc']
        
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    
    def save_model(self, model_path=None):
        """
        Save the trained model.
        
        Args:
            model_path: Path to save the model
        """
        if model_path is None:
            model_path = config.MODEL_PATH
        
        # Create model directory if it doesn't exist
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Save model
        torch.save(self.model, model_path)
        
        # Save class names alongside the model
        classes_file = os.path.join(os.path.dirname(model_path), 'classes.json')
        with open(classes_file, 'w') as f:
            json.dump(self.classes, f)
        
        print(f"Model saved to {model_path}")
        print(f"Classes saved to {classes_file}")
    
    def save_history(self, history_path=None):
        """
        Save training history to a JSON file.
        
        Args:
            history_path: Path to save the history
        """
        if history_path is None:
            history_path = os.path.join(self.output_dir, 'training_history.json')
        
        # Convert numpy arrays to lists if necessary
        history_serializable = {}
        for key, value in self.history.items():
            if isinstance(value, np.ndarray):
                history_serializable[key] = value.tolist()
            else:
                history_serializable[key] = value
        
        # Save history
        with open(history_path, 'w') as f:
            json.dump(history_serializable, f)
        
        print(f"Training history saved to {history_path}")
    
    def plot_history(self, save_path=None):
        """
        Plot and optionally save training history.
        
        Args:
            save_path: Path to save the plot
        """
        if save_path is None:
            save_path = os.path.join(self.output_dir, 'training_plot.png')
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot loss
        ax1.plot(self.history['train_loss'], label='Train Loss')
        ax1.plot(self.history['val_loss'], label='Validation Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Plot accuracy
        ax2.plot(self.history['train_acc'], label='Train Accuracy')
        ax2.plot(self.history['val_acc'], label='Validation Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True)
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        
        print(f"Training plot saved to {save_path}")
    
    def evaluate(self, test_path=None, confusion_matrix=True):
        """
        Evaluate the model on a test set and optionally generate a confusion matrix.
        
        Args:
            test_path: Path to test data (defaults to validation set)
            confusion_matrix: Whether to generate a confusion matrix
            
        Returns:
            metrics: Dictionary of evaluation metrics
        """
        # If no test path provided, use validation set
        if test_path is None and self.val_loader is not None:
            test_loader = self.val_loader
        else:
            # Create test dataset and loader
            test_transform = get_transforms(train=False)
            test_dataset = CustomImageDataset(test_path, transform=test_transform)
            test_loader = DataLoader(
                test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=4,
                pin_memory=True if self.device.type == 'cuda' else False
            )
        
        # Switch to evaluation mode
        self.model.eval()
        
        # Tracking variables
        all_predictions = []
        all_labels = []
        correct = 0
        total = 0
        
        # Evaluate without computing gradients
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs, 1)
                
                # Track statistics
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # Store for confusion matrix
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        accuracy = 100 * correct / total
        print(f"Test Accuracy: {accuracy:.2f}%")
        
        # Generate confusion matrix
        if confusion_matrix:
            self._plot_confusion_matrix(all_labels, all_predictions)
        
        return {'accuracy': accuracy}
    
    def _plot_confusion_matrix(self, y_true, y_pred):
        """
        Generate and plot a confusion matrix.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
        """
        from sklearn.metrics import confusion_matrix
        import seaborn as sns
        
        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Plot
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=self.classes,
            yticklabels=self.classes
        )
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        
        # Save
        plt.tight_layout()
        cm_path = os.path.join(self.output_dir, 'confusion_matrix.png')
        plt.savefig(cm_path)
        plt.close()
        
        print(f"Confusion matrix saved to {cm_path}")


def main():
    """Main function to demonstrate model training."""
    print("Neural Network Model Trainer")
    print("===========================")
    
    # Create training data directory if it doesn't exist
    os.makedirs(config.TRAINING_DATA_PATH, exist_ok=True)
    
    # Check if training data exists
    if not os.path.exists(config.TRAINING_DATA_PATH) or not os.listdir(config.TRAINING_DATA_PATH):
        print(f"Error: Training data directory '{config.TRAINING_DATA_PATH}' is empty or doesn't exist.")
        print("Please add training data before running the trainer.")
        return
    
    # Initialize trainer
    trainer = ModelTrainer()
    
    # Prepare data
    trainer.prepare_data()
    
    # Train model
    trainer.train()
    
    # Evaluate model
    trainer.evaluate()
    
    print("\nTraining and evaluation complete. The model is saved and ready to use!")


if __name__ == "__main__":
    main() 