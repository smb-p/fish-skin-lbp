import os
import cv2

def preprocess_images(input_dir, output_dir, target_size=(256, 256)):
    """
    Resizes all images in the input directory and saves them to the output directory.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for disease_folder in os.listdir(input_dir):
        disease_input_dir = os.path.join(input_dir, disease_folder)
        disease_output_dir = os.path.join(output_dir, disease_folder)
        
        if not os.path.exists(disease_input_dir):
            print(f"Skipping missing folder: {disease_input_dir}")
            continue
        
        os.makedirs(disease_output_dir, exist_ok=True)
        
        for filename in os.listdir(disease_input_dir):
            if filename.endswith((".jpg", ".png")):
                image_path = os.path.join(disease_input_dir, filename)
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                
                if image is None:
                    print(f"Error reading {image_path}. Skipping...")
                    continue
                
                resized_image = cv2.resize(image, target_size)
                normalized_image = resized_image / 255.0  # Normalize to [0, 1]
                
                output_path = os.path.join(disease_output_dir, filename)
                cv2.imwrite(output_path, (normalized_image * 255).astype("uint8"))

# Define paths
base_input_dir = "D:/research paper/dataset"
base_output_dir = "D:/research paper/results/preprocessed"

# Preprocess train and test data
preprocess_images(os.path.join(base_input_dir, "train"), os.path.join(base_output_dir, "train"))
preprocess_images(os.path.join(base_input_dir, "test"), os.path.join(base_output_dir, "test"))