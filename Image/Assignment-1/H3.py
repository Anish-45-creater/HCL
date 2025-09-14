import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Base directory for MVTec AD dataset (pill category for tablet defects)
base_dir = './archive/pill/'

# Chosen image: defective pill with scratch
image_path = os.path.join(base_dir, 'test', 'scratch', '000.png')
ground_truth_path = os.path.join(base_dir, 'ground_truth', 'scratch', '000_mask.png')

# Load image in grayscale
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
if img is None:
    raise ValueError(f"Image not found at {image_path}")

# Load ground truth mask if available
ground_truth = cv2.imread(ground_truth_path, cv2.IMREAD_GRAYSCALE)
if ground_truth is not None:
    _, ground_truth = cv2.threshold(ground_truth, 127, 255, cv2.THRESH_BINARY)

# Preprocess: Apply contrast stretching and Gaussian blur
img = cv2.convertScaleAbs(img, alpha=1.5, beta=0)  # Contrast stretching
img = cv2.GaussianBlur(img, (5, 5), 0)  # Reduce noise

# Debug: Plot histogram to check intensity distribution
plt.figure(figsize=(8, 4))
plt.hist(img.ravel(), bins=256, range=[0, 255], color='gray')
plt.title('Image Histogram')
plt.xlabel('Intensity')
plt.ylabel('Frequency')
plt.savefig('image_histogram.png')
plt.show()

# Otsu Thresholding for defect segmentation
_, otsu_mask = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Adaptive Thresholding for defect segmentation
adaptive_mask = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

# Visualization
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
axes[0].imshow(img, cmap='gray')
axes[0].set_title('Preprocessed Image')
axes[0].axis('off')

axes[1].imshow(otsu_mask, cmap='gray')
axes[1].set_title('Otsu Thresholding')
axes[1].axis('off')

axes[2].imshow(adaptive_mask, cmap='gray')
axes[2].set_title('Adaptive Thresholding')
axes[2].axis('off')

if ground_truth is not None:
    axes[3].imshow(ground_truth, cmap='gray')
    axes[3].set_title('Ground Truth Mask')
    axes[3].axis('off')

plt.tight_layout()
plt.savefig('defect_segmentation.png')
plt.show()

# Save segmented masks
cv2.imwrite('otsu_mask.png', otsu_mask)
cv2.imwrite('adaptive_mask.png', adaptive_mask)
print("Segmented masks saved as 'otsu_mask.png' and 'adaptive_mask.png'")
print("Visualization saved as 'defect_segmentation.png'")