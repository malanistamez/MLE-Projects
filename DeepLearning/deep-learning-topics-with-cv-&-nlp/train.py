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
import smdebug.pytorch as smd

# Suppress truncated image loading warnings
ImageFile.LOAD_TRUNCATED_IMAGES = True

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

def test(model, loader, criterion, hook):
    """
    Function to test the trained model on the test dataset.

    Args:
        model (torch.nn.Module): The trained model.
        loader (torch.utils.data.DataLoader): DataLoader for the test dataset.
        criterion: The loss function.
        hook: Debugger hook.

    Returns:
        None
    """
    model.eval()
    hook.set_mode(smd.modes.EVAL)
    test_loss = 0
    running_corrects = 0
    
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        test_loss += criterion(outputs, labels).item()
        _, preds = torch.max(outputs, 1)
        running_corrects += torch.sum(preds==labels.data).item()
        
    average_accuracy = running_corrects / len(loader.dataset)
    average_loss = test_loss / len(loader.dataset)
    logger.info(f'Test set: Average loss: {average_loss}, Average accuracy: {100*average_accuracy}%')

def train(model, train_loader, valid_loader, epochs, criterion, optimizer, hook):
    """
    Function to train the model.

    Args:
        model (torch.nn.Module): The model to be trained.
        train_loader (torch.utils.data.DataLoader): DataLoader for the training dataset.
        valid_loader (torch.utils.data.DataLoader): DataLoader for the validation dataset.
        epochs (int): Number of epochs for training.
        criterion: The loss function.
        optimizer: The optimizer.
        hook: Debugger hook.

    Returns:
        model (torch.nn.Module): The trained model.
    """
    count = 0
    
    for epoch in range(epochs):
        hook.set_mode(smd.modes.TRAIN)
        model.train()
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            count += len(inputs)
            
        hook.set_mode(smd.modes.EVAL)
        model.eval()
        running_corrects = 0
        
        with torch.no_grad():
            for inputs, labels in valid_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)
                running_corrects += torch.sum(preds == labels.data).item()
                
        total_accuracy = running_corrects / len(valid_loader.dataset)
        logger.info(f'Validation set: Average accuracy: {100*total_accuracy}%')
        
    return model    

def create_data_loaders(data_dir, batch_size):
    """
    Function to create data loaders for training, validation, and testing.

    Args:
        data_dir (str): Path to the data directory.
        batch_size (int): Batch size for data loaders.

    Returns:
        train_loader (torch.utils.data.DataLoader): DataLoader for the training dataset.
        test_loader (torch.utils.data.DataLoader): DataLoader for the testing dataset.
        valid_loader (torch.utils.data.DataLoader): DataLoader for the validation dataset.
    """
    train_path = os.path.join(data_dir, 'train')
    test_path = os.path.join(data_dir, 'test')
    valid_path = os.path.join(data_dir, 'valid')
    
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
    valid_dataset = torchvision.datasets.ImageFolder(root=valid_path, transform=test_transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader, valid_loader

def main(args):
    """
    Main function to train and test the model.

    Args:
        args: Command-line arguments.

    Returns:
        None
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = net()
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss() 
    optimizer = optim.Adam(model.fc.parameters(), lr=0.001)
    
    hook = smd.Hook.create_from_json_file()
    hook.register_hook(model)
    
    train_loader, test_loader, valid_loader = create_data_loaders(args.data, args.batch_size)
    
    model = train(model, train_loader, valid_loader, args.epochs, criterion, optimizer, hook)
    
    test(model, test_loader, criterion, hook)
    
    logger.info("Saving the model")
    torch.save(model.state_dict(), os.path.join(args.model, 'model.pth'))
    logger.info("Model saved")

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    
    # Set up command-line arguments
    parser.add_argument("--batch_size", type=int, default=64, metavar="N", help="input batch size for training")
    parser.add_argument("--epochs", type=int, default=10, metavar="N", help="number of epochs to train")
    parser.add_argument("--data", type=str, default=os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train"), help="training data path")
    parser.add_argument("--model", type=str, default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"), help="output model path")
    
    args = parser.parse_args()
    
    # Call the main function
    main(args)
