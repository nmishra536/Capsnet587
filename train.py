import os
import numpy as np
import tensorflow as tf
from data_loader import PrescriptionDataLoader
from capsnet_model import CapsNet, margin_loss
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (ModelCheckpoint, 
                                      EarlyStopping, 
                                      ReduceLROnPlateau,
                                      CSVLogger)
import matplotlib.pyplot as plt

config = {
    'img_size': (32, 32),
    'batch_size': 16,
    'epochs': 100,
    'routings': 3,
    'lr': 0.001,
    'save_dir': 'results',
    'model_name': 'prescription_capsnet'
}

def train(config):
    os.makedirs(config['save_dir'], exist_ok=True)

    data_loader = PrescriptionDataLoader(img_size=config['img_size'])
    (X_train, y_train), (X_val, y_val), (X_test, y_test), class_names = data_loader.load_data()
    
    print(f"\nDataset Info:")
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Testing samples: {len(X_test)}")
    print(f"Number of classes: {len(class_names)}")
    print(f"Class names: {class_names}\n")

    input_shape = (config['img_size'][0], config['img_size'][1], 1)
    train_model, eval_model = CapsNet(
        input_shape=input_shape,
        n_class=len(class_names),
        routings=config['routings']
    )
    
    train_model.summary()  # Verify model architecture
    
    train_model.compile(
        optimizer=Adam(learning_rate=config['lr']),
        loss=[margin_loss, 'mse'],
        loss_weights=[1., 0.392],
        metrics={'capsnet': 'accuracy'}
    )

    callbacks = [
        CSVLogger(os.path.join(config['save_dir'], f"{config['model_name']}_log.csv")),
        ModelCheckpoint(
            os.path.join(config['save_dir'], f"{config['model_name']}_weights.h5"),
            monitor='val_capsnet_accuracy',
            save_best_only=True,
            save_weights_only=True,
            verbose=1
        ),
        EarlyStopping(
            monitor='val_capsnet_accuracy',
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_capsnet_accuracy',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
    ]

    print("Starting training...\n")
    history = train_model.fit(
        [X_train, y_train],
        [y_train, X_train],
        batch_size=config['batch_size'],
        epochs=config['epochs'],
        validation_data=([X_val, y_val], [y_val, X_val]),
        callbacks=callbacks,
        verbose=2
    )

    eval_model.save_weights(os.path.join(config['save_dir'], f"{config['model_name']}_final.h5"))
    print("\nTraining complete. Model weights saved.")

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['capsnet_accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_capsnet_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.savefig(os.path.join(config['save_dir'], 'training_history.png'))
    plt.close()

    y_pred, _ = eval_model.predict(X_test)
    test_acc = np.mean(np.argmax(y_pred, 1) == np.argmax(y_test, 1))
    print(f"\nTest Accuracy: {test_acc:.4f}")

if __name__ == "__main__":
    train(config)