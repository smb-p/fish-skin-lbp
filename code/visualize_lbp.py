from skimage.feature import local_binary_pattern
import matplotlib.pyplot as plt
import cv2

def visualize_lbp(image_path, radius=3, n_points=8):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Unable to read image: {image_path}")
    
    # Compute LBP
    lbp = local_binary_pattern(image, n_points, radius, method="uniform")
    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap="gray")
    plt.title("Original Image")
    plt.axis("off")
    
    plt.subplot(1, 2, 2)
    plt.imshow(lbp, cmap="gray")
    plt.title("LBP Texture Patterns")
    plt.axis("off")
    
    plt.show()

# Visualize LBP for a test image
img_path = r"C:\Users\sonof\Downloads\EUS  (2).jpg"
visualize_lbp(img_path)