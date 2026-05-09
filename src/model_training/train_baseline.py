import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from  tqdm.notebook import tqdm
from dataset import split_dataset,split_dataset_by_LOSO
from metrics import check_best_model_accuracy,plot_results
from model import LSA64Classifier
# Paths
DATA_DIR = "data/raw"
POSE_DIR = "data/poses/raw"
METADATA_DIR = "data/metadata/real_labels.csv"





def train_baseline():
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(device)
    val_metrics, test_metrics = [], []
    
    for signer in range(10):  
        train_loader, val_loader, test_loader = split_dataset_by_LOSO(signer+1)  
        model,best_model_state = train_epochs(device, train_loader, val_loader)
        signer_val_metrics, signer_test_metrics = check_best_model_accuracy(device, val_loader, test_loader, best_model_state)
        val_metrics.append(signer_val_metrics)
        test_metrics.append(signer_test_metrics)
        print(f"\n✓ Signer {signer} completado:")
        print(f"  → Validation Acc: {signer_val_metrics['accuracy']:.2f}%")
        print(f"  → Test Acc: {signer_test_metrics['accuracy']:.2f}%")
    plot_results(val_metrics, test_metrics, num_signers=10)

def save_model(model):
    MODEL_PATH = "lsa64_baseline_lstm.pth"
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Modelo guardado exitosamente en: {MODEL_PATH}")

def train_epochs(device, train_loader, val_loader):
    model = LSA64Classifier().to(device)
    num_epochs = 200
    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    best_model_state = None 
    not_improving = 0 
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=8,threshold=0.01)
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    #optimizer, T_0=10, T_mult=2, eta_min=1e-6
    #)

    print(f"Starting training on {device} with Dropout enabled...")

    for epoch in range(num_epochs):
        train_loss = train_one_epoch(device, train_loader, model, criterion, optimizer, epoch)
        train_losses.append(train_loss)

        val_loss = validate_one_epoch(device, val_loader, model, criterion, epoch)
        val_losses.append(val_loss)
        scheduler.step(val_loss)
        # Early stopping logic
        if val_loss < best_val_loss:
          not_improving = 0
          best_val_loss = val_loss
          best_model_state = model.state_dict().copy() # Save the best model state
          print(f"Validation loss improved. Saving model state.")
        else:
          not_improving +=1
          if not_improving ==16:
            print(f"Validation loss did not improve. Stopping early.")
            break

        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")

    # Load the best model state before checking accuracy
    if best_model_state:
      model.load_state_dict(best_model_state)
      print("Loaded best model state for evaluation.")
    return model,best_model_state

def validate_one_epoch(device, val_loader, model, criterion, epoch):
    model.eval()
    running_val_loss = 0.0
    with torch.no_grad():
        for sign, labels,lengths in tqdm(val_loader, desc=f'Epoch {epoch+1} Validation'):
          sign, labels = sign.to(device), labels.to(device)
          outputs = model(sign,lengths)
          loss = criterion(outputs, labels)
          running_val_loss += loss.item() * sign.size(0)

    val_loss = running_val_loss / len(val_loader.dataset)
    return val_loss

def train_one_epoch(device, train_loader, model, criterion, optimizer, epoch):
    model.train()
    running_loss = 0.0
    for images, labels,lengths in tqdm(train_loader, desc=f'Epoch {epoch+1} Training'):
      images, labels = images.to(device), labels.to(device)
      optimizer.zero_grad()
      outputs = model(images,lengths)
      loss = criterion(outputs, labels)
      loss.backward()
      optimizer.step()
      running_loss += loss.item() * images.size(0)

    train_loss = running_loss / len(train_loader.dataset)
    return train_loss


train_baseline()
