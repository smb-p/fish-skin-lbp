import os
import numpy as np
import cv2
from skimage.feature import local_binary_pattern
import joblib

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

def predict_disease(image_path, model_path, label_map):
    """
    Predicts the disease category for a given image.
    """
    model = joblib.load(model_path)
    lbp_features = extract_lbp_features(image_path)
    predicted_label = model.predict([lbp_features])[0]
    predicted_label = int(predicted_label)  # Convert np.int64 to int
    
    reverse_label_map = {v: k for k, v in label_map.items()}
    
    # Validate the predicted label
    if predicted_label not in reverse_label_map:
        raise ValueError(f"Unknown label: {predicted_label}")
    
    return reverse_label_map[predicted_label]

# Define paths and label map
model_path = r"D:\research paper\results\predictions\fish_disease_model_xgboost.pkl"
label_map = {
    "Argulus": 0,
    "Broken antennae and rostrum": 1,
    "EUS": 2,
    "Red Spot": 3,
    "Tail And Fin Rot": 4,
    "THE BACTERIAL GILL ROT": 5
}

# Test multiple images
test_images = [
    r"C:\Users\sonof\Downloads\EUS  (2).jpg"]

for img_path in test_images:
    if not os.path.exists(img_path):
        print(f"File does not exist: {img_path}")
        continue
    
    try:
        predicted_disease = predict_disease(img_path, model_path, label_map)
        print(f"Image: {img_path}\nPredicted Disease: {predicted_disease}\n")
    except Exception as e:
        print(f"Error processing image: {img_path}\nError: {e}\n")