import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

base_dir = './archive/pill/'
mask_path = os.path.join(base_dir, 'ground_truth', 'crack', '000_mask.png')  # Changed to 'hole' for potential holes

mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
if mask is None:
    raise ValueError(f"Mask not found at {mask_path}. Ensure MVTec AD 'pill' category is downloaded and path is correct.")

# Debug: Check original mask properties
print(f"Original mask min: {mask.min()}, max: {mask.max()}")
print(f"Unique values in original mask: {np.unique(mask)}")

_, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

# Debug: Check binary mask properties
print(f"Unique values in binary mask: {np.unique(binary_mask)}")

# Define structuring elements (larger kernels for more effect)
open_kernel = np.ones((9, 9), np.uint8)  # Larger kernel for opening
close_kernel = np.ones((11, 11), np.uint8)  # Larger kernel for closing

# Apply morphological opening (removes small noise)
opened_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, open_kernel, iterations=3)

# Apply morphological closing (closes small holes)
closed_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, close_kernel, iterations=3)

# Visualization
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
axes[0].imshow(binary_mask, cmap='gray')
axes[0].set_title('Original Binary Mask')
axes[0].axis('off')

axes[1].imshow(opened_mask, cmap='gray')
axes[1].set_title('After Opening')
axes[1].axis('off')

axes[2].imshow(closed_mask, cmap='gray')
axes[2].set_title('After Closing')
axes[2].axis('off')

plt.tight_layout()
plt.savefig('mask_morphological_processing.png')
plt.show()

# Save processed masks
cv2.imwrite('opened_mask.png', opened_mask)
cv2.imwrite('closed_mask.png', closed_mask)
print("Processed masks saved as 'opened_mask.png' and 'closed_mask.png'")