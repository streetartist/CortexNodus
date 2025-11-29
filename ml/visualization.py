import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
import os

def save_plot(fig, path):
    """Save matplotlib figure to file"""
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fig.savefig(path, bbox_inches='tight', dpi=100)
        plt.close(fig)
        return True
    except Exception as e:
        print(f"Error saving plot to {path}: {e}")
        plt.close(fig)
        return False

def plot_confusion_matrix(y_true, y_pred, classes, save_path):
    """
    Plot confusion matrix using seaborn heatmap
    y_true: list or array of true labels
    y_pred: list or array of predicted labels
    classes: list of class names
    """
    from sklearn.metrics import confusion_matrix
    
    try:
        cm = confusion_matrix(y_true, y_pred)
        
        # Normalize
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_norm = np.nan_to_num(cm_norm)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', 
                    xticklabels=classes, yticklabels=classes, ax=ax)
        
        ax.set_xlabel('Predicted labels')
        ax.set_ylabel('True labels')
        ax.set_title('Confusion Matrix (Normalized)')
        
        return save_plot(fig, save_path)
    except Exception as e:
        print(f"Error plotting confusion matrix: {e}")
        return False

def plot_predictions(images, labels, preds, classes, save_path, num_samples=16):
    """
    Plot a grid of sample predictions
    images: tensor of shape (N, C, H, W)
    labels: tensor of true labels
    preds: tensor of predicted labels
    """
    try:
        # Unnormalize images if needed (assuming standard normalization)
        # For simplicity, just clip to 0-1 for display
        
        fig = plt.figure(figsize=(12, 12))
        
        count = min(len(images), num_samples)
        rows = int(np.ceil(np.sqrt(count)))
        cols = int(np.ceil(count / rows))
        
        for i in range(count):
            ax = fig.add_subplot(rows, cols, i + 1)
            
            img = images[i].cpu().numpy()
            if img.shape[0] == 1: # Grayscale
                img = img.squeeze(0)
                cmap = 'gray'
            else:
                img = np.transpose(img, (1, 2, 0))
                # Simple denormalization for visualization
                img = (img - img.min()) / (img.max() - img.min())
                cmap = None
                
            ax.imshow(img, cmap=cmap)
            
            true_label = classes[labels[i]] if classes else str(labels[i].item())
            pred_label = classes[preds[i]] if classes else str(preds[i].item())
            
            color = 'green' if labels[i] == preds[i] else 'red'
            
            ax.set_title(f"T: {true_label}\nP: {pred_label}", color=color, fontsize=9)
            ax.axis('off')
            
        plt.tight_layout()
        return save_plot(fig, save_path)
    except Exception as e:
        print(f"Error plotting predictions: {e}")
        return False

def plot_loss_curve(history, save_path):
    """
    Plot training and validation loss
    history: list of dicts with 'epoch', 'loss', 'val_loss' (optional)
    """
    try:
        epochs = [h['epoch'] for h in history]
        loss = [h['loss'] for h in history]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(epochs, loss, label='Training Loss', marker='o')
        
        if 'val_loss' in history[0]:
            val_loss = [h['val_loss'] for h in history]
            ax.plot(epochs, val_loss, label='Validation Loss', marker='x')
            
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training Loss Curve')
        ax.legend()
        ax.grid(True)
        
        return save_plot(fig, save_path)
    except Exception as e:
        print(f"Error plotting loss curve: {e}")
        return False
