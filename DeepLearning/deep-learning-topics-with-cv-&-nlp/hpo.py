import argparse
import logging
import os
import sys
from PIL import ImageFile
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms

# Suppress truncated image loading warnings
ImageFile.LOAD_TRUNCATED_IMAGES = True

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

def test(model, loader, criterion):
    """
    Function to test the trained model on the test dataset.

    Args:
        model (torch.nn.Module): The trained model.
        loader (torch.utils.data.DataLoader): DataLoader for the test dataset.
        criterion: The loss function.

    Returns:
        None
    """
    model.eval()
    test_loss = 0
    running_corrects = 0
    
    for inputs, labels in loader:
        outputs = model(inputs)
        test_loss += criterion(outputs, labels).item()
        _, preds = torch.max(outputs, 1)
        running_corrects += torch.sum(preds == labels.data).item()
    
    average_accuracy = running_corrects / len(loader.dataset)
    average_loss = test_loss / len(loader.dataset)
    logger.info(f'Test set: Average loss: {average_loss}, Accuracy: {100 * average_accuracy}%')

def train(model, loader, epochs, criterion, optimizer): 
    """
    Function to train the model.

    Args:
        model (torch.nn.Module): The model to be trained.
        loader (torch.utils.data.DataLoader): DataLoader for the training dataset.
        epochs (int): Number of epochs for training.
        criterion: The loss function.
        optimizer: The optimizer.

    Returns:
        model (torch.nn.Module): The trained model.
    """
    count = 0
    
    for epoch in range(epochs):
        model.train()
        
        for inputs, labels in loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            count += len(inputs)
            if count > 400:
                break
            
    return model 
    
def net():
    """
    Function to define the neural network architecture.

    Returns:
        model (torch.nn.Module): The neural network model.
    """
    model = models.resnet50(pretrained=True)
    
    for param in model.parameters():
        param.requires_grad = False 
    
    num_features = model.fc.in_features
    num_classes = 133
    model.fc = nn.Sequential(
        nn.Linear(num_features, 256), 
        nn.ReLU(),                 
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, num_classes),
        nn.LogSoftmax(dim=1)
    )
    return model

def create_data_loaders(data, batch_size):
    """
    Function to create data loaders for training, validation, and testing.

    Args:
        data (str): Path to the data directory.
        batch_size (int): Batch size for data loaders.

    Returns:
        train_loader (torch.utils.data.DataLoader): DataLoader for the training dataset.
        test_loader (torch.utils.data.DataLoader): DataLoader for the testing dataset.
        valid_loader (torch.utils.data.DataLoader): DataLoader for the validation dataset.
    """
    train_path = os.path.join(data, 'train')
    test_path = os.path.join(data, 'test')
    validation_path = os.path.join(data, 'valid')
    
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.Resize(256),
        transforms.Resize((224, 224)),
        transforms.ToTensor()])

    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.Resize((224, 224)),
        transforms.ToTensor()])
    
    train_dataset = torchvision.datasets.ImageFolder(root=train_path, transform=train_transform)    
    test_dataset = torchvision.datasets.ImageFolder(root=test_path, transform=test_transform)
    validation_dataset = torchvision.datasets.ImageFolder(root=validation_path, transform=test_transform)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader, validation_loader

def main(args):
    """
    Main function to train and test the model.

    Args:
        args: Command-line arguments.

    Returns:
        None
    """
    model = net()
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=0.001)
    
    train_loader, test_loader, _ = create_data_loaders(data=args.data_dir, batch_size=args.batch_size)
    
    model = train(model, train_loader, args.epochs, criterion, optimizer)
    
    test(model, test_loader, criterion)
    
    logger.info("Saving the model")
    torch.save(model.state_dict(), os.path.join(args.model_dir, 'model.pth'))
    logger.info("Model saved")

    
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    
    # Set up command-line arguments
    parser.add_argument("--batch_size", type=int, default=64, metavar="N", help="input batch size for training")
    parser.add_argument("--epochs", type=int, default=10, metavar="N", help="number of epochs to train")
    parser.add_argument("--data_dir", type=str, default=os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train"), help="training data path in S3")
    parser.add_argument("--model_dir", type=str, default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"), help="location to save the model to")
    
    args = parser.parse_args()
    
    # Call the main function
    main(args)
