import os
import numpy as np
from PIL import Image
import pandas as pd
from tensorflow.keras.utils import to_categorical

class PrescriptionDataLoader:
    def __init__(self, dataset_dir="Doctor’s Handwritten Prescription BD dataset", img_size=(32, 32)):
        """
        Args:
            dataset_dir: Name of your dataset folder (sibling to these scripts)
            img_size: Target image size (height, width)
        """
        self.dataset_dir = dataset_dir
        self.img_size = img_size
        self.class_names = []
        self.num_classes = 0

    def _load_split(self, split_name):
        """Load images and labels from a split directory"""
        words_dir = os.path.join(self.dataset_dir, split_name, f"{split_name.lower()}_words")
        labels_path = os.path.join(self.dataset_dir, split_name, f"{split_name.lower()}_labels.csv")
        
        # Read CSV file
        labels_df = pd.read_csv(labels_path)
        print(f"\nLoading data from {labels_path}")
        print("Sample data:")
        print(labels_df.head(3))
        
        # Verify required columns exist
        if 'IMAGE' not in labels_df.columns:
            raise ValueError(f"CSV missing 'IMAGE' column. Found columns: {list(labels_df.columns)}")
        
        # Use MEDICINE_NAME if available, otherwise GENERIC_NAME
        label_col = 'MEDICINE_NAME' if 'MEDICINE_NAME' in labels_df.columns else 'GENERIC_NAME'
        print(f"Using label column: {label_col}")
        
        images = []
        labels = []
        
        for _, row in labels_df.iterrows():
            # Try multiple image extensions
            base_name = str(row['IMAGE']).split('.')[0]  # Remove extension if present
            img_path = None
            for ext in ['.png', '.jpg', '.jpeg', '']:
                test_path = os.path.join(words_dir, f"{base_name}{ext}")
                if os.path.exists(test_path):
                    img_path = test_path
                    break
            
            if not img_path:
                print(f"Image not found for {base_name} in {words_dir}")
                continue
                
            try:
                img = Image.open(img_path).convert('L')  # Convert to grayscale
                img = img.resize(self.img_size)
                images.append(np.array(img) / 255.0)  # Normalize to [0,1]
                labels.append(row[label_col])
            except Exception as e:
                print(f"Skipping {img_path}: {str(e)}")
                continue
        
        return np.array(images), np.array(labels)

    def load_data(self):
        # Load all splits
        X_train, y_train = self._load_split("Training")
        X_val, y_val = self._load_split("Validation")
        X_test, y_test = self._load_split("Testing")
        
        # Get unique class names from training set
        self.class_names = sorted(np.unique(y_train))
        self.num_classes = len(self.class_names)
        class_to_idx = {cls: i for i, cls in enumerate(self.class_names)}
        
        # Convert labels to indices and one-hot encode
        y_train = to_categorical([class_to_idx[cls] for cls in y_train], self.num_classes)
        y_val = to_categorical([class_to_idx[cls] for cls in y_val], self.num_classes)
        y_test = to_categorical([class_to_idx[cls] for cls in y_test], self.num_classes)
        
        # Add channel dimension (grayscale)
        X_train = np.expand_dims(X_train, axis=-1)
        X_val = np.expand_dims(X_val, axis=-1)
        X_test = np.expand_dims(X_test, axis=-1)
        
        return (X_train, y_train), (X_val, y_val), (X_test, y_test), self.class_names