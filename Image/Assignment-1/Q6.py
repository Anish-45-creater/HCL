import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Set base directory and image paths
base_dir = './archive/pill/'
image_path = os.path.join(base_dir, 'test', 'crack', '003.png')
mask_path = os.path.join(base_dir, 'ground_truth', 'crack', '003_mask.png')

# Load image and mask
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
if img is None or mask is None:
    raise ValueError(f"Image or mask not found at {image_path} or {mask_path}. Ensure MVTec AD 'pill' category is downloaded and paths are correct.")

# Preprocessing
# Initial binary image using a simple threshold (can be tuned)
_, binary = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY_INV)

# Morphological closing to fill small gaps
kernel = np.ones((5, 5), np.uint8)
binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=3)

# Combine binary image with mask to focus on defect regions
binary = cv2.bitwise_and(binary, binary, mask=mask)

# Connected Component Labeling
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)

# Classification using mask overlap
defect_components = []
min_area = 50  # Lower minimum to detect small defect parts
max_area = 5000  # Maximum area for larger defects

for i in range(1, num_labels):  # Skip background
    area = stats[i, cv2.CC_STAT_AREA]
    x = stats[i, cv2.CC_STAT_LEFT]
    y = stats[i, cv2.CC_STAT_TOP]
    w = stats[i, cv2.CC_STAT_WIDTH]
    h = stats[i, cv2.CC_STAT_HEIGHT]
    bbox = (x, y, w, h)
    
    # Check overlap with mask
    roi_mask = mask[y:y+h, x:x+w]
    mask_area = np.sum(roi_mask > 0)  # Number of white pixels in mask region
    overlap_ratio = mask_area / area if area > 0 else 0
    
    # Consider as defect if it overlaps significantly with mask and meets area criteria
    if overlap_ratio > 0.5 and min_area < area < max_area:
        defect_components.append((i, bbox, area))

# Merge nearby defect components
merged_defects = []
while defect_components:
    current_label, current_bbox, current_area = defect_components.pop(0)
    x, y, w, h = current_bbox
    merged_bbox = [x, y, x + w, y + h]  # [min_x, min_y, max_x, max_y]
    merged_area = current_area
    i = 0
    while i < len(defect_components):
        label, bbox, area = defect_components[i]
        bx, by, bw, bh = bbox
        bx2, by2 = bx + bw, by + bh
        # Merge if components are within 50 pixels
        if (abs(bx - merged_bbox[0]) < 50 or abs(bx2 - merged_bbox[2]) < 50) and \
           (abs(by - merged_bbox[1]) < 50 or abs(by2 - merged_bbox[3]) < 50):
            merged_bbox = [
                min(merged_bbox[0], bx),
                min(merged_bbox[1], by),
                max(merged_bbox[2], bx2),
                max(merged_bbox[3], by2)
            ]
            merged_area += area
            defect_components.pop(i)
        else:
            i += 1
    merged_defects.append((current_label, (merged_bbox[0], merged_bbox[1], merged_bbox[2] - merged_bbox[0], merged_bbox[3] - merged_bbox[1]), merged_area))

# Visualization
output_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
for label, bbox, area in merged_defects:
    x, y, w, h = bbox
    cv2.rectangle(output_img, (x, y), (x + w, y + h), (0, 0, 255), 2)
    cv2.putText(output_img, f'Defect (Area: {area})', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

# Overlay mask for reference (green outline)
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(output_img, contours, -1, (0, 255, 0), 1)

fig, axes = plt.subplots(1, 2, figsize=(18, 6))
axes[0].imshow(img, cmap='gray')
axes[0].set_title('Original Tablet Image')
axes[0].axis('off')


axes[1].imshow(cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB))
axes[1].set_title('Detected Defects with Bounding Boxes (Red) and Mask Outline (Green)')
axes[1].axis('off')

plt.tight_layout()
plt.savefig('tablet_defect_detection.png')
plt.show()

print(f"Total components found: {num_labels - 1}")
print(f"Defect components: {len(merged_defects)}")
if len(merged_defects) > 0:
    print("Defects detected: Regions overlapping with mask highlighted in red.")
else:
    print("No defects detected. Check mask alignment or preprocessing parameters.")

cv2.imwrite('tablet_defects_output.png', output_img)
print("Output image saved as 'tablet_defects_output.png'")