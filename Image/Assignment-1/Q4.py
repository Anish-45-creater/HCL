import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import precision_recall_fscore_support

# Base directory for the MVTec AD dataset
base_dir = './archive/screw/'

# Sample defective images and their ground truth masks
image_paths = [
    os.path.join(base_dir, 'test', 'scratch_head', '000.png'),  # Scratch on head
    os.path.join(base_dir, 'test', 'scratch_neck', '001.png'),  # Scratch on neck
    os.path.join(base_dir, 'test', 'thread_side', '002.png')    # Defect on thread side
]

mask_paths = [
    os.path.join(base_dir, 'ground_truth', 'scratch_head', '000_mask.png'),
    os.path.join(base_dir, 'ground_truth', 'scratch_neck', '001_mask.png'),
    os.path.join(base_dir, 'ground_truth', 'thread_side', '002_mask.png')
]

# Function to apply thresholding methods
def apply_thresholding(img):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 1. Global Thresholding (fixed threshold of 120, inverted for defects as white)
    _, global_thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY_INV)
    
    # 2. Adaptive Thresholding (Gaussian method)
    adaptive_thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                           cv2.THRESH_BINARY_INV, 11, 2)
    
    # 3. Otsu Thresholding (automatic threshold)
    _, otsu_thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    return gray, global_thresh, adaptive_thresh, otsu_thresh

# Function to compute IoU
def compute_iou(pred_mask, true_mask):
    intersection = np.logical_and(pred_mask, true_mask).sum()
    union = np.logical_or(pred_mask, true_mask).sum()
    return intersection / union if union != 0 else 0

# Process each image
for idx, (img_path, mask_path) in enumerate(zip(image_paths, mask_paths)):
    # Load image and ground truth mask
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error: Image not found at {img_path}")
        continue
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"Error: Mask not found at {mask_path}")
        continue
    mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)[1]  # Ensure binary

    # Apply thresholding
    original_gray, global_thresh, adaptive_thresh, otsu_thresh = apply_thresholding(img)

    # Flatten masks for metrics (1D arrays)
    mask_flat = (mask / 255).astype(int).flatten()
    global_flat = (global_thresh / 255).astype(int).flatten()
    adaptive_flat = (adaptive_thresh / 255).astype(int).flatten()
    otsu_flat = (otsu_thresh / 255).astype(int).flatten()

    # Compute metrics
    global_metrics = precision_recall_fscore_support(mask_flat, global_flat, average='binary')
    adaptive_metrics = precision_recall_fscore_support(mask_flat, adaptive_flat, average='binary')
    otsu_metrics = precision_recall_fscore_support(mask_flat, otsu_flat, average='binary')
    global_iou = compute_iou(global_thresh, mask)
    adaptive_iou = compute_iou(adaptive_thresh, mask)
    otsu_iou = compute_iou(otsu_thresh, mask)

    # Visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()

    axes[0].imshow(original_gray, cmap='gray')
    axes[0].set_title('Original Gray')
    axes[0].axis('off')

    axes[1].imshow(global_thresh, cmap='gray')
    axes[1].set_title(f'Global (IoU: {global_iou:.3f}, F1: {global_metrics[2]:.3f})')
    axes[1].axis('off')

    axes[2].imshow(adaptive_thresh, cmap='gray')
    axes[2].set_title(f'Adaptive (IoU: {adaptive_iou:.3f}, F1: {adaptive_metrics[2]:.3f})')
    axes[2].axis('off')

    axes[3].imshow(otsu_thresh, cmap='gray')
    axes[3].set_title(f'Otsu (IoU: {otsu_iou:.3f}, F1: {otsu_metrics[2]:.3f})')
    axes[3].axis('off')

    axes[4].imshow(mask, cmap='gray')
    axes[4].set_title('Ground Truth Mask')
    axes[4].axis('off')

    axes[5].axis('off')  # Placeholder

    plt.suptitle(f'Thresholding on Screw Image {idx+1}')
    plt.tight_layout()
    plt.show()

    # Print detailed metrics
    print(f"\nResults for Image {idx+1} ({os.path.basename(img_path)}):")
    print(f"Global Thresholding - IoU: {global_iou:.3f}, Precision: {global_metrics[0]:.3f}, "
          f"Recall: {global_metrics[1]:.3f}, F1: {global_metrics[2]:.3f}")
    print(f"Adaptive Thresholding - IoU: {adaptive_iou:.3f}, Precision: {adaptive_metrics[0]:.3f}, "
          f"Recall: {adaptive_metrics[1]:.3f}, F1: {adaptive_metrics[2]:.3f}")
    print(f"Otsu Thresholding - IoU: {otsu_iou:.3f}, Precision: {otsu_metrics[0]:.3f}, "
          f"Recall: {otsu_metrics[1]:.3f}, F1: {otsu_metrics[2]:.3f}")

