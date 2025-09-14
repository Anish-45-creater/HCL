import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Path to MVTec AD dataset (adjust based on your directory)
dataset_path = './archive/screw/test/scratch_head/000.png'  # Example image with defect
mask_path = './archive/screw/ground_truth/scratch_head/000_mask.png'  # Corresponding mask

# Load the original image (color or grayscale; using grayscale for edge detection)
original = cv2.imread(dataset_path)
if original is None:
    raise ValueError(f"Image not found at {dataset_path}. Ensure path is correct.")
original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)

# Load ground truth mask (binary mask of defects; 255 for defect, 0 for background)
mask = None
if os.path.exists(mask_path):
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)[1]  # Ensure binary
else:
    print("Ground truth mask not found. Proceeding without overlay.")

# 1. Gaussian Blur for noise reduction and surface smoothing
kernel_size = (5, 5)  # Adjustable kernel size
sigma = 1.0  # Standard deviation for Gaussian
smoothed = cv2.GaussianBlur(original_gray, kernel_size, sigma)

# 2. Sobel Edge Detection (highlights scratches as horizontal/vertical edges)
sobel_x = cv2.Sobel(smoothed, cv2.CV_64F, 1, 0, ksize=3)  # Horizontal edges
sobel_y = cv2.Sobel(smoothed, cv2.CV_64F, 0, 1, ksize=3)  # Vertical edges
sobel_combined = np.sqrt(sobel_x**2 + sobel_y**2)
sobel_combined = cv2.convertScaleAbs(sobel_combined)  # Convert back to uint8

# 3. Laplacian Filter (detects fine edges and surface transitions)
laplacian = cv2.Laplacian(smoothed, cv2.CV_64F, ksize=3)
laplacian = cv2.convertScaleAbs(np.uint8(np.absolute(laplacian)))

# 4. Unsharp Masking (enhances details by subtracting blurred version)
gaussian_blur = cv2.GaussianBlur(original_gray, (0, 0), 2.0)
unsharp = cv2.addWeighted(original_gray, 1.5, gaussian_blur, -0.5, 0)

# Optional: Combine filters (e.g., multiply Sobel and Laplacian for stronger defect highlighting)
enhanced = cv2.multiply(sobel_combined, laplacian) // 255
enhanced = cv2.convertScaleAbs(enhanced)

# Visualization
fig, axes = plt.subplots(2, 3, figsize=(15, 10))  # 2x3 grid = 6 subplots
axes = axes.ravel()

# Original
axes[0].imshow(original_gray, cmap='gray')
axes[0].set_title('Original Image')
axes[0].axis('off')

# Smoothed
axes[1].imshow(smoothed, cmap='gray')
axes[1].set_title('Gaussian Blur (Surface Enhancement)')
axes[1].axis('off')

# Sobel
axes[2].imshow(sobel_combined, cmap='gray')
axes[2].set_title('Sobel Edges (Scratches/Dents)')
axes[2].axis('off')

# Laplacian
axes[3].imshow(laplacian, cmap='gray')
axes[3].set_title('Laplacian (Fine Details)')
axes[3].axis('off')

# Unsharp
axes[4].imshow(unsharp, cmap='gray')
axes[4].set_title('Unsharp Masking (Detail Enhancement)')
axes[4].axis('off')

# Enhanced Combined
axes[5].imshow(enhanced, cmap='gray')
axes[5].set_title('Combined Enhancement (Scratches/Dents Highlighted)')
axes[5].axis('off')

# Display Ground Truth Mask and Overlay in a separate figure if mask exists
if mask is not None:
    fig_mask, (ax_mask, ax_overlay) = plt.subplots(1, 2, figsize=(10, 5))
    
    # Ground Truth Mask
    ax_mask.imshow(mask, cmap='gray')
    ax_mask.set_title('Ground Truth Defect Mask')
    ax_mask.axis('off')
    
    # Overlay on Original
    overlay = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    overlay[mask == 255] = [255, 0, 0]  # Highlight defects in red
    ax_overlay.imshow(overlay)
    ax_overlay.set_title('Original with Defect Overlay')
    ax_overlay.axis('off')
    
    plt.tight_layout()
    plt.show()



# Save enhanced image
cv2.imwrite('enhanced_metal_surface.png', enhanced)
print("Enhanced image saved as 'enhanced_metal_surface.png'")

# Quantitative: Compute edge strength in defect areas (if mask available)
if mask is not None:
    defect_pixels = enhanced[mask == 255]
    if len(defect_pixels) > 0:
        mean_defect_intensity = np.mean(defect_pixels)
        print(f"Mean intensity in defect areas (highlighted scratches/dents): {mean_defect_intensity:.2f}")
    else:
        print("No defect pixels found.")