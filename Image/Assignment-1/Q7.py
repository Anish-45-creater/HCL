import cv2
import numpy as np
import os
import glob
import json
from collections import defaultdict
import pandas as pd  # For summary table

def load_annotations(labeled_path, image_filename):
    """
    Load ground truth from labeled folder. Assumes COCO-style JSON or per-image JSON.
    Returns dict of {'good': [bboxes], 'defective': [bboxes]} or empty if not found.
    """
    # Try per-image JSON
    json_path = os.path.join(labeled_path, f"{os.path.splitext(image_filename)[0]}.json")
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            annot = json.load(f)
        # Assume annot has 'categories' and 'annotations' or direct 'joints' list
        good_bboxes = []
        defective_bboxes = []
        for ann in annot.get('annotations', []):
            category = ann.get('category_id', 0)
            if category == 1:  # Assume 1=good, 2=defective; adjust based on dataset
                good_bboxes.append(ann['bbox'])
            elif category == 2:
                defective_bboxes.append(ann['bbox'])
        return {'good': good_bboxes, 'defective': defective_bboxes}
    
    # Fallback: Global COCO JSON
    global_json = os.path.join(labeled_path, 'annotations.json')  # Adjust filename if different
    if os.path.exists(global_json):
        with open(global_json, 'r') as f:
            coco = json.load(f)
        image_id = next((img['id'] for img in coco['images'] if img['file_name'] == image_filename), None)
        if image_id:
            good_bboxes = [ann['bbox'] for ann in coco['annotations'] if ann['image_id'] == image_id and ann['category_id'] == 1]
            defective_bboxes = [ann['bbox'] for ann in coco['annotations'] if ann['image_id'] == image_id and ann['category_id'] == 2]
            return {'good': good_bboxes, 'defective': defective_bboxes}
    
    return {'good': [], 'defective': []}

def compute_iou(boxA, boxB):
    """IoU for validation against ground truth."""
    if not boxA or not boxB:
        return 0.0
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    union = areaA + areaB - inter
    return inter / union if union > 0 else 0.0

def classify_solder_joint(stats, label):
    """
    Classify based on features. Tune thresholds for SOLDEF_AI (e.g., small SMT joints).
    Good: Compact, circular (area 100-500, extent >0.6, circularity >0.7)
    Defective: Irregular (low extent, low circularity, or extreme area).
    """
    area = stats[label, cv2.CC_STAT_AREA]
    left, top, width, height = stats[label, cv2.CC_STAT_LEFT], stats[label, cv2.CC_STAT_TOP], stats[label, cv2.CC_STAT_WIDTH], stats[label, cv2.CC_STAT_HEIGHT]
    bbox = [left, top, left + width, top + height]
    extent = area / (width * height)
    # Approximate perimeter (OpenCV stats may not have it; use contour if needed)
    perimeter_approx = np.sqrt(4 * np.pi * area)  # Circle approximation
    circularity = (4 * np.pi * area) / (perimeter_approx ** 2) if perimeter_approx > 0 else 0
    
    # Tuned for small solder joints in dataset
    if 100 < area < 500 and extent > 0.6 and circularity > 0.7:
        return 'good', bbox
    else:
        return 'defective', bbox

def process_pcb_image(img_path, labeled_path):
    """
    Process image: CCL on binarized image, classify, validate with labels.
    Returns detected_good, detected_defective, gt_good_count, gt_defective_count, matches.
    """
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return [], [], 0, 0, 0
    
    # Preprocess: Blur and adaptive threshold (solder joints are bright blobs)
    blurred = cv2.GaussianBlur(img, (3, 3), 0)  # Smaller kernel for small joints
    binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    # If solder is dark, invert: binary = cv2.bitwise_not(binary)
    
    # CCL
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8, ltype=cv2.CV_32S)
    
    detected_good = []
    detected_defective = []
    matches = 0  # IoU > 0.5 matches
    
    filename = os.path.basename(img_path)
    gt = load_annotations(labeled_path, filename)
    gt_good_count = len(gt['good'])
    gt_defective_count = len(gt['defective'])
    
    for i in range(1, num_labels):  # Skip background
        cls, bbox = classify_solder_joint(stats, i)
        if cls == 'good':
            detected_good.append(bbox)
            # Validate against GT good
            for gt_box in gt['good']:
                if compute_iou(bbox, gt_box) > 0.5:
                    matches += 1
                    break
        else:
            detected_defective.append(bbox)
            # Validate against GT defective
            for gt_box in gt['defective']:
                if compute_iou(bbox, gt_box) > 0.5:
                    matches += 1
                    break
    
    return detected_good, detected_defective, gt_good_count, gt_defective_count, matches

def main():
    dataset_path = './SOLDEF_AI/Dataset/'  # Raw images in Dataset/
    labeled_path = './SOLDEF_AI/Labeled/'  # Annotations in Labeled/
    
    if not os.path.exists(dataset_path):
        print(f"Dataset not found at {dataset_path}. Ensure unzipped correctly.")
        return
    
    # Recursively find all images in Dataset/ subfolders (CS1, RB805, etc.)
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(dataset_path, '**', ext), recursive=True))
    
    if not image_paths:
        print("No images found in Dataset/. Check subfolders like CS1, RB805.")
        return
    
    print(f"Processing {len(image_paths)} images from {len([p for p in image_paths if 'CS' in p])} CS* and other model folders.")
    
    total_detected_good = 0
    total_detected_defective = 0
    total_gt_good = 0
    total_gt_defective = 0
    total_matches = 0
    image_stats = []
    
    model_stats = defaultdict(lambda: {'good': 0, 'defective': 0, 'images': 0})
    
    for img_path in image_paths:
        detected_good, detected_defective, gt_good, gt_defective, matches = process_pcb_image(img_path, labeled_path)
        num_d_good = len(detected_good)
        num_d_defective = len(detected_defective)
        total_detected_good += num_d_good
        total_detected_defective += num_d_defective
        total_gt_good += gt_good
        total_gt_defective += gt_defective
        total_matches += matches
        
        # Extract model (e.g., CS1, RB805 from path)
        model = next((f for f in ['CS1','CS2','CS3','CS4','CS5','CS6','CS7','RB805','R1206'] if f in img_path), 'Other')
        model_stats[model]['good'] += num_d_good
        model_stats[model]['defective'] += num_d_defective
        model_stats[model]['images'] += 1
        
        image_name = os.path.basename(img_path)
        image_stats.append({
            'image': image_name,
            'model': model,
            'detected_good': num_d_good,
            'detected_defective': num_d_defective,
            'gt_good': gt_good,
            'gt_defective': gt_defective,
            'matches': matches
        })
        
        # Save labeled image (optional)
        if detected_good or detected_defective:
            img_color = cv2.imread(img_path)
            if img_color is not None:
                # Draw detected bboxes (green good, red defective)
                for bbox in detected_good:
                    cv2.rectangle(img_color, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
                for bbox in detected_defective:
                    cv2.rectangle(img_color, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 0, 255), 2)
                output_path = img_path.replace('.', '_detected.')
                cv2.imwrite(output_path, img_color)
    
    # Overall Statistics
    total_detected = total_detected_good + total_detected_defective
    detected_good_pct = (total_detected_good / total_detected * 100) if total_detected > 0 else 0
    detected_defective_pct = (total_detected_defective / total_detected * 100) if total_detected > 0 else 0
    
    total_gt = total_gt_good + total_gt_defective
    gt_good_pct = (total_gt_good / total_gt * 100) if total_gt > 0 else 0
    gt_defective_pct = (total_gt_defective / total_gt * 100) if total_gt > 0 else 0
    
    precision = (total_matches / total_detected * 100) if total_detected > 0 else 0  # Approx precision
    
    print("\n=== Overall Statistics ===")
    print(f"Total images processed: {len(image_paths)}")
    print(f"Total detected solder joints: {total_detected}")
    print(f"Detected good: {total_detected_good} ({detected_good_pct:.2f}%)")
    print(f"Detected defective: {total_detected_defective} ({detected_defective_pct:.2f}%)")
    print(f"Ground truth good: {total_gt_good} ({gt_good_pct:.2f}%)")
    print(f"Ground truth defective: {total_gt_defective} ({gt_defective_pct:.2f}%)")
    print(f"Matching detections (IoU>0.5): {total_matches} (approx. precision: {precision:.2f}%)")
    
    # Per-model stats
    print("\n=== Per-Model Statistics ===")
    for model, stats in model_stats.items():
        if stats['images'] > 0:
            good_pct = (stats['good'] / (stats['good'] + stats['defective']) * 100)
            print(f"{model}: {stats['images']} images, {stats['good']} good, {stats['defective']} defective ({good_pct:.2f}% good)")
    
    # Summary table (first 10 images)
    print("\n=== Per-Image Statistics (first 10) ===")
    df_summary = pd.DataFrame(image_stats[:10])
    print(df_summary.to_string(index=False))
    
    # Save full stats to CSV
    pd.DataFrame(image_stats).to_csv('soldef_stats.csv', index=False)
    print("\nFull stats saved to soldef_stats.csv")

if __name__ == "__main__":
    main()