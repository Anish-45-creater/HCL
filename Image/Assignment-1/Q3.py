import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Base directory for MVTec AD dataset (carpet category)
base_dir = './archive/carpet/'

# Sample defective fabric image and its ground truth mask
image_path = os.path.join(base_dir, 'test', 'cut', '000.png')  # Defective image (e.g., cut defect)
mask_path = os.path.join(base_dir, 'ground_truth', 'cut', '000_mask.png')  # Ground truth (optional for validation)

# Load image in grayscale
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
if img is None:
    raise ValueError(f"Image not found at {image_path}. Ensure MVTec AD 'carpet' is downloaded.")

# Load ground truth mask if available (binary: 255 for defect)
ground_truth = None
if os.path.exists(mask_path):
    ground_truth = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    ground_truth = cv2.threshold(ground_truth, 127, 255, cv2.THRESH_BINARY)[1]

# Step 1: Binarize the image using adaptive thresholding (handles fabric texture variations)
binary = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

# Step 2: Morphological Opening (remove small noise/objects)
kernel = np.ones((5, 5), np.uint8)  # Kernel size tunable (larger for bigger noise removal)
opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

# Step 3: Morphological Closing (fill small holes/gaps in defects)
closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

# Step 4: Isolate defective regions (final mask after operations)
isolated_defects = closing  # This is the isolated binary mask

# Visualization
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.ravel()

# Original
axes[0].imshow(img, cmap='gray')
axes[0].set_title('Original Fabric Image')
axes[0].axis('off')

# Binary Thresholded
axes[1].imshow(binary, cmap='gray')
axes[1].set_title('Binary Thresholded')
axes[1].axis('off')

# After Opening
axes[2].imshow(opening, cmap='gray')
axes[2].set_title('After Opening (Noise Removal)')
axes[2].axis('off')

# After Closing
axes[3].imshow(closing, cmap='gray')
axes[3].set_title('After Closing (Gap Filling)')
axes[3].axis('off')

# Isolated Defects Overlay on Original
overlay = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
overlay[isolated_defects == 255] = [255, 0, 0]  # Highlight defects in red
axes[4].imshow(overlay)
axes[4].set_title('Isolated Defects (Red Overlay)')
axes[4].axis('off')

# Ground Truth (if available)
if ground_truth is not None:
    axes[5].imshow(ground_truth, cmap='gray')
    axes[5].set_title('Ground Truth Mask')
else:
    axes[5].text(0.5, 0.5, 'No Ground Truth Available', ha='center', va='center')
axes[5].axis('off')

plt.tight_layout()
plt.show()

# Save isolated defects mask
cv2.imwrite('isolated_defects_mask.png', isolated_defects)
print("Isolated defects mask saved as 'isolated_defects_mask.png'")

# Optional: Compute accuracy if ground truth available
if ground_truth is not None:
    accuracy = np.mean(isolated_defects == ground_truth) * 100
    print(f"Accuracy compared to ground truth: {accuracy:.2f}%")