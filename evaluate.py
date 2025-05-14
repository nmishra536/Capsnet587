import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (classification_report, 
                           confusion_matrix)
import seaborn as sns
from data_loader import PrescriptionDataLoader
from capsnet_model import CapsNet

# Configuration (must match training config)
config = {
    'img_size': (32, 32),
    'save_dir': 'results',
    'model_name': 'prescription_capsnet'
}

def evaluate(config):
    # Load data
    print("Loading data...")
    data_loader = PrescriptionDataLoader(img_size=config['img_size'])
    (_, _), (_, _), (X_test, y_test), class_names = data_loader.load_data()
    
    # Build evaluation model
    print("Building model...")
    input_shape = (config['img_size'][0], config['img_size'][1], 1)
    _, eval_model = CapsNet(
        input_shape=input_shape,
        n_class=len(class_names),
        routings=3
    )
    
    # Load weights
    model_path = os.path.join(config['save_dir'], f"{config['model_name']}_weights.h5")
    if os.path.exists(model_path):
        eval_model.load_weights(model_path)
        print(f"Loaded weights from {model_path}")
    else:
        raise FileNotFoundError(f"Model weights not found at {model_path}")

    # Evaluate
    print("\nRunning evaluation...")
    y_pred, X_recon = eval_model.predict(X_test)
    
    # Convert predictions
    y_true = np.argmax(y_test, axis=1)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(
        y_true, 
        y_pred_classes, 
        target_names=class_names,
        digits=4
    ))
    
    # Confusion matrix
    plt.figure(figsize=(12, 10))
    cm = confusion_matrix(y_true, y_pred_classes)
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(config['save_dir'], 'confusion_matrix.png'))
    plt.close()
    print("Saved confusion matrix to results/confusion_matrix.png")
    
    # Sample reconstructions
    n_samples = 5
    indices = np.random.choice(len(X_test), n_samples, replace=False)
    
    plt.figure(figsize=(n_samples * 3, 6))
    for i, idx in enumerate(indices):
        # Original image
        plt.subplot(2, n_samples, i + 1)
        plt.imshow(X_test[idx].squeeze(), cmap='gray')
        plt.title(f"True: {class_names[y_true[idx]]}\nPred: {class_names[y_pred_classes[idx]]}")
        plt.axis('off')
        
        # Reconstructed image
        plt.subplot(2, n_samples, i + 1 + n_samples)
        plt.imshow(X_recon[idx].squeeze(), cmap='gray')
        plt.axis('off')
    
    plt.savefig(os.path.join(config['save_dir'], 'reconstructions.png'))
    plt.close()
    print("Saved reconstructions to results/reconstructions.png")
    
    # Save predictions
    results_df = pd.DataFrame({
        'True_Class': [class_names[i] for i in y_true],
        'Predicted_Class': [class_names[i] for i in y_pred_classes],
        'Correct': y_true == y_pred_classes
    })
    results_df.to_csv(os.path.join(config['save_dir'], 'predictions.csv'), index=False)
    print("Saved detailed predictions to results/predictions.csv")

if __name__ == "__main__":
    evaluate(config)