import os
import numpy as np
from skimage.feature import local_binary_pattern
import cv2

def extract_lbp_features(image_path, radius=3, n_points=8):
    """
    Extracts Local Binary Pattern (LBP) features from an image.
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Unable to read image: {image_path}")
    
    # Compute LBP
    lbp = local_binary_pattern(image, n_points, radius, method="uniform")
    
    # Histogram of LBP features
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)  # Normalize histogram
    return hist

def prepare_dataset(dataset_dir):
    features = []
    labels = []
    label_map = {}  # Map folder names to numeric labels
    
    for label_idx, disease_folder in enumerate(os.listdir(dataset_dir)):
        label_map[disease_folder] = label_idx
        disease_dir = os.path.join(dataset_dir, disease_folder)
        
        for filename in os.listdir(disease_dir):
            if filename.endswith((".jpg", ".png")):
                image_path = os.path.join(disease_dir, filename)
                lbp_features = extract_lbp_features(image_path)
                features.append(lbp_features)
                labels.append(label_map[disease_folder])
    
    return np.array(features), np.array(labels), label_map

# Define paths
train_dir = "D:/research paper/results/preprocessed/train"
test_dir = "D:/research paper/results/preprocessed/test"

# Prepare dataset
X_train, y_train, label_map = prepare_dataset(train_dir)
X_test, y_test, _ = prepare_dataset(test_dir)

# Save features and labels
features_dir = "D:/research paper/results/features"
os.makedirs(features_dir, exist_ok=True)

np.save(os.path.join(features_dir, "train_features.npy"), X_train)
np.save(os.path.join(features_dir, "train_labels.npy"), y_train)
np.save(os.path.join(features_dir, "test_features.npy"), X_test)
np.save(os.path.join(features_dir, "test_labels.npy"), y_test)

print("Label Map:", label_map)