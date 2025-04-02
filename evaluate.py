from ultralytics import YOLO
import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def plot_confusion_matrix(model_path, data_path):
    """Plot confusion matrix for model evaluation"""
    # Load model
    model = YOLO(model_path)
    
    # Run validation with confusion matrix
    results = model.val(data=data_path, conf=0.25, iou=0.6, verbose=True)
    
    # Get confusion matrix
    conf_matrix = results.confusion_matrix.matrix
    
    # Class names
    class_names = list(results.names.values())
    
    # Plot confusion matrix
    plt.figure(figsize=(12, 10))
    
    # Create normalized confusion matrix
    conf_matrix_normalized = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
    
    sns.heatmap(conf_matrix_normalized, annot=True, cmap="Blues", fmt=".2f",
                xticklabels=class_names, yticklabels=class_names)
    
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Normalized Confusion Matrix')
    plt.tight_layout()
    
    # Save figure
    plt.savefig(os.path.join('runs', 'detect', 'confusion_matrix.png'))
    plt.show()
    
    return results

if __name__ == "__main__":
    import numpy as np
    
    # Path to your best model
    model_path = os.path.join("runs", "train2", "weights", "best.pt")
    
    # Path to your data YAML
    data_path = os.path.join('data', 'PCB.yaml')
    
    # Evaluate model and plot confusion matrix
    results = plot_confusion_matrix(model_path, data_path)
    
    # Print key metrics
    print("\nPer-class Metrics:")
    for i, class_name in results.names.items():
        print(f"Class {class_name}:")
        print(f"  Precision: {results.precision[i]:.4f}")
        print(f"  Recall: {results.recall[i]:.4f}")
        print(f"  mAP50: {results.maps[i]:.4f}")
        print(f"  mAP50-95: {results.map50_95[i]:.4f}")
    
    print("\nOverall Metrics:")
    print(f"Overall mAP50: {results.map50:.4f}")
    print(f"Overall mAP50-95: {results.map50_95:.4f}")