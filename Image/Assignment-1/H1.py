import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Base directory for MVTec AD dataset (e.g., 'screw' category for industrial images)
base_dir = './archive/screw/'

# Sample industrial image paths (using defective and good for comparison)
image_paths = [
    os.path.join(base_dir, 'test', 'scratch_head', '000.png'),  # Defective image
    os.path.join(base_dir, 'train', 'good', '000.png')  # Good image
]

# Function to apply Sobel and Canny edge detectors
def apply_edge_detectors(img_path):
    # Load image in grayscale
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error: Image not found at {img_path}")
        return None, None, None
    
    # Preprocess: Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    
    # Sobel Edge Detector
    sobel_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    sobel = cv2.convertScaleAbs(np.sqrt(sobel_x**2 + sobel_y**2))
    
    # Canny Edge Detector
    canny = cv2.Canny(blurred, 50, 150)  # Thresholds tunable
    
    return img, sobel, canny

# Process and visualize for each image
for idx, img_path in enumerate(image_paths):
    original, sobel, canny = apply_edge_detectors(img_path)
    if original is None:
        continue
    
    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(original, cmap='gray')
    axes[0].set_title('Original')
    axes[0].axis('off')
    
    axes[1].imshow(sobel, cmap='gray')
    axes[1].set_title('Sobel Edges')
    axes[1].axis('off')
    
    axes[2].imshow(canny, cmap='gray')
    axes[2].set_title('Canny Edges')
    axes[2].axis('off')
    
    plt.suptitle(f'Edge Detection on Industrial Image {idx+1}')
    plt.tight_layout()
    plt.savefig(f'industrial_edge_detection_{idx+1}.png')
    plt.show()
    
    # Save results
    cv2.imwrite(f'sobel_{idx+1}.jpg', sobel)
    cv2.imwrite(f'canny_{idx+1}.jpg', canny)
    print(f"Results saved for image {idx+1}")