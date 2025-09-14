import os
import numpy as np
import cv2
from skimage import io, measure, morphology
from sklearn.metrics import jaccard_score
import matplotlib.pyplot as plt

def load_bottle_dataset(base_path='./archive/bottle/', max_test_images=3):
    train_path = os.path.join(base_path, 'train/good')
    test_path = os.path.join(base_path, 'test')
    gt_path = os.path.join(base_path, 'ground_truth')

    # Training images (good only)
    train_images = [io.imread(os.path.join(train_path, f)) for f in os.listdir(train_path) if f.endswith('.png')]
    train_gray = [cv2.resize(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), (512, 512)) for img in train_images]
    train_gray = np.array(train_gray)

    # Test images (limit to 3, prioritize defective)
    test_files = []
    test_subdirs = [d for d in os.listdir(test_path) if os.path.isdir(os.path.join(test_path, d))]
    for subdir in test_subdirs:
        if subdir != 'good':
            subdir_path = os.path.join(test_path, subdir)
            files = [f for f in os.listdir(subdir_path) if f.endswith('.png')]
            if files:
                test_files.append((os.path.join(subdir_path, files[0]), subdir, files[0]))
        if len(test_files) >= max_test_images:
            break
    if len(test_files) < max_test_images:
        good_path = os.path.join(test_path, 'good')
        files = [f for f in os.listdir(good_path) if f.endswith('.png')]
        for f in files[:max_test_images - len(test_files)]:
            test_files.append((os.path.join(good_path, f), 'good', f))

    test_images = [io.imread(path) for path, _, _ in test_files]
    test_gray = [cv2.resize(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), (512, 512)) for img in test_images]
    test_gray = np.array(test_gray)
    labels = [subdir == 'good' for _, subdir, _ in test_files]

    # Ground truth masks
    gt_masks = []
    for path, subdir, fname in test_files:
        if subdir != 'good':
            gt_fname = fname.replace('.png', '_mask.png')
            gt_full = os.path.join(gt_path, subdir, gt_fname)
            if os.path.exists(gt_full):
                gt = io.imread(gt_full)
                if gt.ndim == 3:
                    gt = gt[:, :, 0]
                gt = gt > 0
                gt = cv2.resize(gt.astype(np.uint8), (512, 512), interpolation=cv2.INTER_NEAREST)
            else:
                gt = np.zeros((512, 512), dtype=np.uint8)
        else:
            gt = np.zeros((512, 512), dtype=np.uint8)
        gt_masks.append(gt)
    gt_masks = np.array(gt_masks)

    return train_gray, test_images, test_gray, labels, gt_masks, test_files

def compute_stats(train_gray):
    mu = np.mean(train_gray, axis=0)
    sigma = np.std(train_gray, axis=0)
    sigma[sigma < 1e-5] = 1e-5
    return mu, sigma

def localize_defects(test_gray, mu, sigma, threshold=2.5):
    anomaly_maps = []
    for img in test_gray:
        deviation = np.abs(img.astype(float) - mu) / sigma
        anomaly_map = (deviation > threshold).astype(np.uint8)
        kernel = np.ones((5, 5), np.uint8)
        anomaly_map = cv2.morphologyEx(anomaly_map, cv2.MORPH_CLOSE, kernel)
        labeled = measure.label(anomaly_map)
        regions = measure.regionprops(labeled)
        for region in regions:
            if region.area < 100:
                anomaly_map[labeled == region.label] = 0
        anomaly_maps.append(anomaly_map)
    return np.array(anomaly_maps)

def extract_template(defective_image_path, roi=None):
    img = io.imread(defective_image_path)
    img_gray = cv2.resize(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), (512, 512))
    if roi:
        x, y, w, h = roi
        template = img_gray[y:y+h, x:x+w]
    else:
        template = img_gray[231:281, 231:281]
    return cv2.GaussianBlur(template, (5, 5), 0)

def pattern_matching(test_gray, template, threshold=0.8):
    matches = []
    match_images = []
    for img in test_gray:
        img_blur = cv2.GaussianBlur(img, (5, 5), 0)
        result = cv2.matchTemplate(img_blur, template, cv2.TM_CCOEFF_NORMED)
        loc = np.where(result >= threshold)
        match_locations = list(zip(*loc[::-1]))
        match_img = img.copy()
        h, w = template.shape
        for pt in match_locations:
            cv2.rectangle(match_img, pt, (pt[0] + w, pt[1] + h), (255), 2)
        matches.append({'locations': match_locations, 'scores': result[loc]})
        match_images.append(match_img)
    return matches, match_images

def evaluate_iou(pred_masks, gt_masks, labels):
    ious = []
    defective_idx = np.where(~np.array(labels))[0]
    for i in defective_idx:
        if np.any(gt_masks[i]):
            pred = pred_masks[i].ravel()
            gt = gt_masks[i].ravel()  # Fixed: Use gt_masks instead of gt
            iou = jaccard_score(gt, pred, pos_label=1, zero_division=0)
            ious.append(iou)
    return np.mean(ious) if ious else 0, defective_idx

# Run pipeline
output_dir = './bottle_outputs/'
os.makedirs(output_dir, exist_ok=True)

train_gray, test_images, test_gray, labels, gt_masks, test_files = load_bottle_dataset()
mu, sigma = compute_stats(train_gray)
pred_masks = localize_defects(test_gray, mu, sigma)
iou, defective_idx = evaluate_iou(pred_masks, gt_masks, labels)
print(f'Mean IoU for defective images: {iou:.4f}')

# Pattern matching
template_path = test_files[0][0] if not labels[0] else './archive/bottle/test/broken_large/000.png'
template = extract_template(template_path)
matches, match_images = pattern_matching(test_gray, template)

# Display images
fig, axes = plt.subplots(3, 3, figsize=(15, 10))
for i, (path, subdir, fname) in enumerate(test_files):
    axes[i, 0].imshow(test_images[i])
    axes[i, 0].set_title(f'Original: {subdir}/{fname}')
    axes[i, 0].axis('off')
    axes[i, 1].imshow(pred_masks[i], cmap='gray')
    axes[i, 1].set_title('Anomaly Map')
    axes[i, 1].axis('off')
    axes[i, 2].imshow(match_images[i], cmap='gray')
    axes[i, 2].set_title(f'Matches: {len(matches[i]["locations"])}')
    axes[i, 2].axis('off')
    # Save images
    cv2.imwrite(os.path.join(output_dir, f'anomaly_{subdir}_{fname}'), pred_masks[i] * 255)
    cv2.imwrite(os.path.join(output_dir, f'match_{subdir}_{fname}'), match_images[i])
plt.tight_layout()
plt.show()

# Print match results
for i, (path, subdir, fname) in enumerate(test_files):
    if matches[i]['locations']:
        print(f'Matches in {subdir}/{fname}: {len(matches[i]["locations"])} locations')