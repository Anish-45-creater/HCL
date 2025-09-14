import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Base directory for the dataset (adjust if needed)
base_dir = './archive/tile/test/'

# Sample cracked tile images (replace with actual paths or use a loop)
image_paths = [
    os.path.join(base_dir, 'crack/000.png'),  # Cracked tile image 1
    os.path.join(base_dir, 'crack/001.png'),  # Cracked tile image 2
    os.path.join(base_dir, 'crack/002.png')   # Cracked tile image 3
]

# Function to apply edge detectors
def apply_edge_detectors(img_path):
    # Load image in grayscale
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error: Image not found at {img_path}")
        return None, None, None, None
    
    # Preprocess: Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    
    # 1. Sobel Edge Detector (computes gradients in x and y)
    sobel_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    sobel = np.sqrt(sobel_x**2 + sobel_y**2)
    sobel = cv2.convertScaleAbs(sobel)
    
    # 2. Laplacian Edge Detector (second derivative for edge sharpening)
    laplacian = cv2.Laplacian(blurred, cv2.CV_64F, ksize=3)
    laplacian = cv2.convertScaleAbs(laplacian)
    
    # 3. Canny Edge Detector (multi-stage: gradient, non-max suppression, hysteresis)
    canny = cv2.Canny(blurred, 50, 150)  # Thresholds tunable based on image
    
    return img, sobel, laplacian, canny

# Process and display for each image
for idx, img_path in enumerate(image_paths):
    original, sobel, laplacian, canny = apply_edge_detectors(img_path)
    if original is None:
        continue
    
    # Visualization
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(original, cmap='gray')
    axes[0].set_title('Original')
    axes[0].axis('off')
    
    axes[1].imshow(sobel, cmap='gray')
    axes[1].set_title('Sobel')
    axes[1].axis('off')
    
    axes[2].imshow(laplacian, cmap='gray')
    axes[2].set_title('Laplacian')
    axes[2].axis('off')
    
    axes[3].imshow(canny, cmap='gray')
    axes[3].set_title('Canny')
    axes[3].axis('off')
    
    plt.suptitle(f'Edge Detection on Cracked Tile Image {idx+1}')
    plt.tight_layout()
    plt.show()
    
    # Save results
    cv2.imwrite(f'sobel_image_{idx+1}.jpg', sobel)
    cv2.imwrite(f'laplacian_image_{idx+1}.jpg', laplacian)
    cv2.imwrite(f'canny_image_{idx+1}.jpg', canny)
    print(f"Results saved for image {idx+1}")