import random
import numpy as np
import torch
from ham10000.models import build_model
import torch.nn as nn
import argparse
from ham10000.utils import load_config
from ham10000.data import build_dataloaders, compute_class_weights
from tqdm import tqdm
import wandb


def set_seed(seed):
    """Seed Python, NumPy, and PyTorch RNGs for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    

def train_one_epoch(model, loader, criterion, optimizer, device):
    """Run one training epoch and return the average training loss."""
    
    model.train()
    running_loss = 0.0
    
    progress = tqdm(loader, desc="train", leave=False)
    for images, labels in progress:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        progress.set_postfix(loss=loss.item())
    
    return running_loss / len(loader.dataset)


def validate(model, loader, criterion, device):
    """Evaluate on the val loader, returning (avg_loss, accuracy)."""
    
    model.eval()
    running_loss = 0.0
    correct = 0
    with torch.no_grad():
        progress = tqdm(loader, desc="validation", leave=False)
        for images, labels in progress:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            
    avg_loss = running_loss / len(loader.dataset)
    accuracy = correct / len(loader.dataset)
    return avg_loss, accuracy


def train(config, train_loader, val_loader):
    """Train the model and save the best checkpoint by validation loss.

    Sets seeds, builds the model/optimizer/loss (with optional class
    weighting per config.use_class_weights), then loops over config.num_epochs,
    logging metrics each epoch and saving the lowest-val-loss model to disk.
    Logs to Weights & Biases when config.use_wandb is set.

    Returns:
        The trained model.
    """
    
    set_seed(config.seed)
    device = torch.device('cuda' if torch.cuda.is_available()
                          else 'mps' if torch.backends.mps.is_available() else 'cpu')
    
    model = build_model(config.num_classes).to(device)
    if config.use_class_weights:
        class_weights = compute_class_weights(train_loader, config.num_classes, device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    
    if config.use_wandb:
        run_name = f"lr{config.learning_rate}_w{config.use_class_weights}_aug{config.use_augmentation}"
        wandb.init(
            project="ham10000-classifier",
            name=run_name,
            config={
                "learning_rate": config.learning_rate,
                "batch_size": config.batch_size,
                "num_epochs": config.num_epochs,
                "use_class_weights": config.use_class_weights,
                "use_augmentation": config.use_augmentation,
                "model_name": config.model_name,
            },
        )
    
    
    best_val_loss = float('inf')
    for epoch in range(config.num_epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        print(f'Epoch {epoch+1}/{config.num_epochs} | '
              f'train_loss={train_loss:.4f} | '
              f'val_loss={val_loss:.4f} | val_acc={val_acc:.4f}')
        
        if config.use_wandb:
            wandb.log({
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "epoch": epoch + 1,
            })
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            print(f'Best model saved with val_loss={val_loss:.4f}')
    
    if config.use_wandb:
        wandb.finish()    
        
    return model

def main():
    parser = argparse.ArgumentParser(description='Train HAM10000 Classifier')
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to the YAML configuration file'
    )
    args = parser.parse_args()
    
    config = load_config(args.config)
    train_loader, val_loader, _ = build_dataloaders(config)
    train(config, train_loader, val_loader)
    

if __name__ == '__main__':
    main()