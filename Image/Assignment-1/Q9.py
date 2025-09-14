import os
import cv2
import numpy as np
from skimage import morphology, io, filters
from skimage.morphology import skeletonize
import matplotlib.pyplot as plt

# Paths (adjust if needed)
DATA_PATH = 'archive/bottle'
TRAIN_PATH = os.path.join(DATA_PATH, 'train/good')
TEST_PATH = os.path.join(DATA_PATH, 'test')
GT_PATH = os.path.join(DATA_PATH, 'ground_truth')

# Step 1: Build reference edge map from training (defect-free)
def build_reference_edge_map(train_path, target_size=(512, 512), num_samples=20):
    edge_maps = []
    samples = [f for f in os.listdir(train_path) if f.endswith('.png')][:num_samples]
    for img_name in samples:
        img_path = os.path.join(train_path, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        img = cv2.resize(img, target_size)
        # Use Sobel for better gradient edges
        sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
        edges = np.hypot(sobelx, sobely)
        edges = cv2.normalize(edges, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        edge_maps.append(edges.astype(np.float32) / 255.0)
    if not edge_maps:
        raise ValueError("No valid training images found.")
    ref_edge = np.mean(edge_maps, axis=0)
    return ref_edge

# Step 2: Process single test image
def detect_defects(img_path, gt_path, ref_edge, target_size=(512, 512), visualize=False):
    # Load image
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Failed to load image: {img_path}")
        return None, None
    
    # Resize to match reference
    img = cv2.resize(img, target_size)
    
    # Load GT if exists
    gt = None
    if gt_path and os.path.exists(gt_path):
        try:
            gt_raw = io.imread(gt_path)
            if gt_raw.ndim == 3:
                gt = (gt_raw[:,:,0] > 0).astype(bool)
            else:
                gt = (gt_raw > 0).astype(bool)
            gt = cv2.resize(gt.astype(np.uint8), target_size) > 0
        except Exception as e:
            print(f"Error loading GT {gt_path}: {e}")
    else:
        print(f"GT not found: {gt_path}")
    
    # Edge detection with Sobel (better for cracks)
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    edges = np.hypot(sobelx, sobely)
    edges = cv2.normalize(edges, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    edges = (edges > filters.threshold_otsu(edges)).astype(np.float32)
    
    # Enhanced Morphological ops
    # Line kernel for cracks (horizontal/vertical)
    line_kernel_h = morphology.rectangle(1, 5)
    line_kernel_v = morphology.rectangle(5, 1)
    line_kernel = np.maximum(line_kernel_h, line_kernel_v)
    
    # Opening with line kernel to remove noise, preserve lines
    edges_morph = morphology.opening(edges, line_kernel)
    # Skeletonize to thin cracks
    edges_skel = skeletonize(edges_morph)
    # Dilation to connect fragments
    edges_morph = morphology.dilation(edges_skel, morphology.disk(1))
    
    # For missing parts: Closing with disk
    edges_closed = morphology.closing(edges_morph, morphology.disk(5))
    
    # Improved Defect map: Local difference (convolve for local mean)
    from scipy.ndimage import convolve
    local_ref = convolve(ref_edge, np.ones((9,9))/81, mode='reflect')  # Local average
    defect_map = np.abs(edges_morph - local_ref)
    defect_map_closed = np.abs(edges_closed - local_ref)
    defect_map = np.maximum(defect_map, defect_map_closed)
    
    # Adaptive threshold for binary (Otsu)
    thresh = filters.threshold_otsu(defect_map[defect_map > 0]) if np.any(defect_map > 0) else 0.05
    binary_defect = defect_map > thresh
    
    # Post-process: Remove small noise
    binary_defect = morphology.remove_small_objects(binary_defect, min_size=50)
    
    # Metrics: IoU if GT available
    iou = None
    if gt is not None:
        intersection = np.logical_and(binary_defect, gt).sum()
        union = np.logical_or(binary_defect, gt).sum()
        iou = intersection / union if union > 0 else 0
    
    # Visualization (enable for debugging)
    if visualize:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes[0,0].imshow(img, cmap='gray'); axes[0,0].set_title('Original')
        axes[0,1].imshow(edges, cmap='gray'); axes[0,1].set_title('Sobel Edges')
        axes[0,2].imshow(edges_morph, cmap='gray'); axes[0,2].set_title('Morph + Skeleton')
        axes[1,0].imshow(defect_map, cmap='hot'); axes[1,0].set_title(f'Defect Map (Thresh: {thresh:.3f})')
        axes[1,1].imshow(binary_defect, cmap='Reds'); axes[1,1].set_title('Binary Defects')
        if gt is not None:
            axes[1,2].imshow(gt, cmap='Greens'); axes[1,2].set_title(f'GT (IoU: {iou:.2f})')
        else:
            axes[1,2].imshow(np.zeros_like(img), cmap='Greens'); axes[1,2].set_title('No GT')
        plt.tight_layout()
        out_name = os.path.basename(img_path).replace('.png', '_vis.png')
        plt.savefig(out_name, dpi=150, bbox_inches='tight')
        plt.show()
    
    return binary_defect, iou

# Main execution
if __name__ == '__main__':
    # Build reference
    ref_edge = build_reference_edge_map(TRAIN_PATH)
    print(f"Reference edge map shape: {ref_edge.shape}")
    
    # Process test defect images
    defect_types = ['broken_small', 'broken_large']
    for defect_type in defect_types:
        test_subpath = os.path.join(TEST_PATH, defect_type)
        if os.path.exists(test_subpath):
            img_files = [f for f in os.listdir(test_subpath) if f.endswith('.png')]
            for img_file in img_files[:5]:
                img_path = os.path.join(test_subpath, img_file)
                gt_file = img_file.replace('.png', '_mask.png')
                gt_path = os.path.join(GT_PATH, defect_type, gt_file)
                binary_defect, iou = detect_defects(img_path, gt_path, ref_edge, visualize=True)  # Set True for first run
                if binary_defect is not None:
                    iou_str = f"{iou:.2f}" if iou is not None else "N/A"
                    print(f"Processed {img_file}: IoU = {iou_str}")
        else:
            print(f"Subfolder not found: {test_subpath}")
    
    print("Pipeline complete. Visualizations saved as *_vis.png")