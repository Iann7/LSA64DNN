import torch
from model import LSA64Classifier
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

PLOTS_DIR = "plots"

def save_current_plot(filename):
    os.makedirs(PLOTS_DIR, exist_ok=True)
    path = os.path.join(PLOTS_DIR, filename)
    plt.savefig(path, dpi=300, bbox_inches='tight')
    print(f"Gráfico guardado en: {path}")

def check_accuracy(loader, model, device):
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y,lengths in tqdm(loader, desc="Checking accuracy"):
            x = x.to(device)
            y = y.to(device)

            scores = model(x,lengths)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

    return float(num_correct) / num_samples



def check_best_model_accuracy(device, val_loader, test_loader, best_model_state):
    val_metrics = get_all_metrics(val_loader, best_model_state, device)
    test_metrics = get_all_metrics(test_loader, best_model_state, device)
    
    print(f"\nValidation - Acc: {val_metrics['accuracy']:.2f}%, F1: {val_metrics['f1']:.4f}")
    print(f"Test - Acc: {test_metrics['accuracy']:.2f}%, F1: {test_metrics['f1']:.4f}")
    
    return val_metrics, test_metrics

def get_all_metrics(loader, model_state, device):
    model = LSA64Classifier().to(device)
    model.load_state_dict(model_state)
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for signs, labels, lengths in loader:
            signs, labels = signs.to(device), labels.to(device)
            outputs = model(signs, lengths)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    metrics = {
        'accuracy': np.mean(all_preds == all_labels) * 100,
        'f1': f1_score(all_labels, all_preds, average='weighted'),
        'precision': precision_score(all_labels, all_preds, average='weighted'),
        'recall': recall_score(all_labels, all_preds, average='weighted'),
        'confusion_matrix': confusion_matrix(all_labels, all_preds),
    }
    
    return metrics

def plot_results(val_metrics_list, test_metrics_list, num_signers):
    """Grafica resultados con matplotlib"""
    
    signers = range(num_signers)
    val_acc = [m['accuracy'] for m in val_metrics_list]
    test_acc = [m['accuracy'] for m in test_metrics_list]
    val_f1 = [m['f1'] * 100 for m in val_metrics_list]
    test_f1 = [m['f1'] * 100 for m in test_metrics_list]
    
    # Figura con 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Gráfico 1: Accuracy
    x = np.arange(num_signers)
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, val_acc, width, label='Validation', color='skyblue', edgecolor='navy', linewidth=0.5)
    bars2 = ax1.bar(x + width/2, test_acc, width, label='Test', color='lightcoral', edgecolor='darkred', linewidth=0.5)
    
    ax1.set_xlabel('Signer (Fold)', fontsize=12)
    ax1.set_ylabel('Accuracy (%)', fontsize=12)
    ax1.set_title('Accuracy por Signer', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'S{i}' for i in signers])
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim([0, 105])
    
    # Añadir valores en las barras
    for bar in bars1:
        height = bar.get_height()
        ax1.annotate(f'{height:.1f}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', fontsize=8)
    for bar in bars2:
        height = bar.get_height()
        ax1.annotate(f'{height:.1f}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', fontsize=8)
    
    # Gráfico 2: F1-Score
    bars3 = ax2.bar(x - width/2, val_f1, width, label='Validation', color='lightgreen', edgecolor='darkgreen', linewidth=0.5)
    bars4 = ax2.bar(x + width/2, test_f1, width, label='Test', color='orange', edgecolor='brown', linewidth=0.5)
    
    ax2.set_xlabel('Signer (Fold)', fontsize=12)
    ax2.set_ylabel('F1-Score (%)', fontsize=12)
    ax2.set_title('F1-Score por Signer', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'S{i}' for i in signers])
    ax2.legend(loc='lower right')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim([0, 105])
    
    for bar in bars3:
        height = bar.get_height()
        ax2.annotate(f'{height:.1f}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', fontsize=8)
    for bar in bars4:
        height = bar.get_height()
        ax2.annotate(f'{height:.1f}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', fontsize=8)
    
    plt.tight_layout()
    save_current_plot("loso_accuracy_f1.png")
    plt.show()
    
    # Estadísticas
    print("\n" + "="*60)
    print("RESUMEN FINAL - LOSO Cross-Validation")
    print("="*60)
    print(f"Validation Accuracy:  {np.mean(val_acc):.2f}% ± {np.std(val_acc):.2f}")
    print(f"Test Accuracy:        {np.mean(test_acc):.2f}% ± {np.std(test_acc):.2f}")
    print(f"Validation F1-Score:  {np.mean(val_f1):.2f}% ± {np.std(val_f1):.2f}")
    print(f"Test F1-Score:        {np.mean(test_f1):.2f}% ± {np.std(test_f1):.2f}")
    print("="*60)

    test_cm = np.sum([m['confusion_matrix'] for m in test_metrics_list], axis=0)
    plot_confusion_matrix(test_cm, title="Matriz de Confusión Agregada - Test LOSO")

def plot_confusion_matrix(cm, title="Matriz de Confusión"):
    """Grafica matriz de confusión"""
    plt.figure(figsize=(14, 12))
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.colorbar()
    plt.title(title, fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    safe_title = title.lower().replace(" ", "_").replace("-", "_")
    save_current_plot(f"{safe_title}.png")
    plt.show()
