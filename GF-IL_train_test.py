import pandas as pd
import numpy as np
import random
import gc

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, roc_auc_score

import GF_IL_model as model


# Load modalities data
marbert_tensor = torch.load('data/marbert-embeddings.pt', weights_only=True)
text_aux_tensor = torch.load('data/modal_text_aux.pt', weights_only=True).to(torch.float32)
user_profile_tensor = torch.load('data/modal_user_profile.pt', weights_only=True).to(torch.float32)
user_cred_tensor = torch.load('data/modal_user_cred.pt', weights_only=True).to(torch.float32)
sentiment_tensor = torch.load('data/modal_sentiment.pt', weights_only=True).to(torch.float32)


# Load corresponding annotated labels
labels_tensor = torch.load('data/labels.pt', weights_only=True)


# Formating function
def format_f(my_float):
    return "{:.2f}%".format(my_float * 100)


# Set random seeds for reproducibility
seed_value = 42

def set_random_seeds(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        

# Hyperparameters
epochs = 100
learn_rates = [0.001, 0.0005, 0.0001]
batch = 64
test_size = 0.2    


# Prepare data for training
def prepare_data_gf_il(tensor_list, labels, test_size, batch_size, random_state):
    train_indices, test_indices = train_test_split(np.arange(len(labels)), test_size=test_size, random_state=random_state, stratify=labels)
    
    # Create training and test datasets
    train_dataset = TensorDataset(tensor_list[0][train_indices], tensor_list[1][train_indices], tensor_list[2][train_indices], tensor_list[3][train_indices], tensor_list[4][train_indices], labels[train_indices])
    test_dataset = TensorDataset(tensor_list[0][test_indices], tensor_list[1][test_indices], tensor_list[2][test_indices], tensor_list[3][test_indices], tensor_list[4][test_indices], labels[test_indices])
    
    torch.manual_seed(random_state)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader


# Train - Test function
def train_test_gf_il(model, train_loader, test_loader, criterion, optimizer, epochs):
    best_val_loss = float('inf')
    # Track metrics for plotting
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    
    # For ROC-AUC calculation
    all_preds = []
    all_labels = []
    
    # Training Loop
    for epoch in range(epochs):
        total_train_loss = 0
        correct_train = 0
        total_train = 0
        model.train()
        
        for x1, x2, x3, x4, x5, labels_train in train_loader:
            # Forward pass
            optimizer.zero_grad()
            outputs = model(x1, x2, x3, x4, x5) 
            batch_loss = criterion(outputs, labels_train)  # Using categorical crossentropy

            # Backward pass and optimization
            batch_loss.backward()
            optimizer.step()
            
            total_train_loss += batch_loss.item()
            _, predicted = torch.max(outputs, 1)
            correct_train += (predicted == labels_train).sum().item()
            total_train += labels_train.size(0)
            
        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        train_accuracy = correct_train / total_train
        train_accuracies.append(train_accuracy)
            
        #------------------------------------------------------
        # Validation Loop
        total_val_loss = 0
        correct_val = 0
        total_val = 0
        model.eval()
        
        with torch.no_grad():
            for x1, x2, x3, x4, x5, labels_test in test_loader:
                outputs = model(x1, x2, x3, x4, x5)
                loss = criterion(outputs, labels_test)
                total_val_loss += loss.item()
                
                _, predicted = torch.max(outputs, 1)
                correct_val += (predicted == labels_test).sum().item()
                total_val += labels_test.size(0)
                
                # Store predictions and labels for ROC-AUC calculation
                all_preds.extend(outputs[:, 1].cpu().numpy())
                all_labels.extend(labels_test.cpu().numpy())

        avg_val_loss = total_val_loss / len(test_loader)
        val_losses.append(avg_val_loss)
        val_accuracy = correct_val / total_val
        val_accuracies.append(val_accuracy)
        
        if epoch % 20 == 0:
            print(f"\nEpoch {epoch+1}, Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")
        else:
            print('*', end='  ')
    
    #----------------------------------------------------------        
    # Evaluate Unseen Test Data
    loss = 0
    accuracy = 0
    all_preds = []
    all_labels = []
    model.eval()
    
    with torch.no_grad():
        for x1, x2, x3, x4, x5, labels_test in test_loader:
            outputs = model(x1, x2, x3, x4, x5)   
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted)
            all_labels.extend(labels_test)

    all_preds = torch.tensor([p.item() for p in all_preds])
    all_labels = torch.tensor([l.item() for l in all_labels])

    # Calculate metrics
    tp = ((all_preds == 1) & (all_labels == 1)).sum().item()
    tn = ((all_preds == 0) & (all_labels == 0)).sum().item()
    fp = ((all_preds == 1) & (all_labels == 0)).sum().item()
    fn = ((all_preds == 0) & (all_labels == 1)).sum().item()
    accuracy_m = accuracy_score(all_labels, all_preds)
    precision_m = precision_score(all_labels, all_preds)
    recall_m = recall_score(all_labels, all_preds)
    f1_m = f1_score(all_labels, all_preds)
    auc_m = roc_auc_score(all_labels, all_preds)

    print('\n', 'Evaluation ...')
    print(f"TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")
    print(f"Accuracy: {accuracy_m:.4f}, Precision: {precision_m:.4f}, Recall: {recall_m:.4f}, F1: {f1_m:.4f}, AUC: {auc_m:.4f}")

    return tp, tn, fp, fn, accuracy_m, precision_m, recall_m, f1_m, auc_m, train_losses, val_losses, train_accuracies, val_accuracies, all_preds, all_labels


# Evaluate GF-IL Model
for lr in learn_rates:
    gc.collect()

    set_random_seeds(seed_value)

    criterion = nn.CrossEntropyLoss()
    model = model.GF_IL()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_loader, test_loader = prepare_data_gf_il([marbert_tensor, text_aux_tensor, user_profile_tensor, user_cred_tensor, sentiment_tensor], labels_tensor, test_size=test_size, batch_size=batch, random_state=seed_value)

    res = train_test_gf_il(model, train_loader, test_loader, criterion, optimizer, epochs=epochs)
    
    printable_results = {'Model': 'GF-IL', 'Hyperparams': lr, 'TP': res[0], 'TN': res[1], 'FP': res[2], 'FN': res[3],'Accuracy': format_f(res[4]), 'Precision': format_f(res[5]), 'Recall': format_f(res[6]), 'F1': format_f(res[7]), 'AUC': format_f(res[8])}
    
    print('-' * 80)
    print(printable_results)