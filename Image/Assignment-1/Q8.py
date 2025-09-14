import cv2
import numpy as np
import pandas as pd
from io import BytesIO
import base64

# Function to compute IoU between two bounding boxes
def compute_iou(boxA, boxB):
    if boxA is None or boxB is None:
        return 0.0
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = interArea / float(boxAArea + boxBArea - interArea) if (boxAArea + boxBArea - interArea) > 0 else 0.0
    return iou

# Function to process uploaded image (assuming it's provided as base64 or file-like)
def process_uploaded_image(image_data):
    img_str = image_data.split(',')[1] if ',' in image_data else image_data
    img_data = base64.b64decode(img_str)
    nparr = np.frombuffer(img_data, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

# Load training annotations (optional, for reference)
train_annot_path = 'flickr_logos_27_dataset/flickr_logos_27_dataset_training_set_annotation.txt'
df_train = pd.read_csv(train_annot_path, sep=r'\s+', header=None, names=['filename', 'class_name', 'subset', 'x1', 'y1', 'x2', 'y2'])

# Select a template from training set (using Adidas as a placeholder, but we'll override)
class_name = 'Adidas'
template_row = df_train[df_train['class_name'] == class_name].iloc[0]
template_img_path = f'flickr_logos_27_dataset/flickr_logos_27_dataset_images/{template_row["filename"]}'
template_img = cv2.imread(template_img_path, cv2.IMREAD_GRAYSCALE)
template = template_img[template_row['y1']:template_row['y2'], template_row['x1']:template_row['x2']]

# Process the query image
query_img_path = './flickr_logos_27_dataset/flickr_logos_27_dataset_images/18526789.jpg'
query_img_color = cv2.imread(query_img_path)
if query_img_color is None:
    print("Error: Could not load query image.")
    exit()
query_img_gray = cv2.cvtColor(query_img_color, cv2.COLOR_BGR2GRAY)
print(f"Query image shape: {query_img_color.shape}")

# Use a portion of the query image as template for self-matching
h, w = query_img_gray.shape
template = query_img_gray[0:int(h/2), 0:int(w/2)]  # Top-left quarter as template

# Preprocess images
query_img_gray = cv2.GaussianBlur(query_img_gray, (5, 5), 0)
query_img_gray = cv2.equalizeHist(query_img_gray)
template = cv2.GaussianBlur(template, (5, 5), 0)
template = cv2.equalizeHist(template)

# --- ORB Feature Matching ---
orb = cv2.ORB_create(nfeatures=1000)
kp_template, des_template = orb.detectAndCompute(template, None)
kp_query, des_query = orb.detectAndCompute(query_img_gray, None)

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des_template, des_query)
matches = sorted(matches, key=lambda x: x.distance)
print(f"Total matches: {len(matches)}")

good_matches = matches[:20]  # Top 20 matches for debugging
orb_bbox = None
orb_good_matches_count = len(good_matches)
if len(good_matches) > 5:
    src_pts = np.float32([kp_template[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp_query[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    
    if M is not None:
        h, w = template.shape
        pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, M)
        dst = np.int32(dst).reshape(4, 2)
        x_min, y_min = np.min(dst, axis=0)
        x_max, y_max = np.max(dst, axis=0)
        orb_bbox = [x_min, y_min, x_max, y_max]
        
        orb_img = query_img_color.copy()
        cv2.rectangle(orb_img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        cv2.imwrite('orb_result.jpg', orb_img)
        print(f"ORB: Logo detected with {orb_good_matches_count} good matches.")
    else:
        print("ORB: No homography found.")
else:
    print(f"ORB: Not enough good matches ({orb_good_matches_count}).")

# --- Template Matching ---
def multi_scale_template_match(image, template, scales=np.linspace(0.3, 2.0, 15)):
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    best_val = -1
    best_loc = None
    best_scale = 1.0
    th, tw = template.shape[:2]
    
    for scale in scales:
        resized_template = cv2.resize(template, (int(tw * scale), int(th * scale)))
        if resized_template.shape[0] > image_gray.shape[0] or resized_template.shape[1] > image_gray.shape[1]:
            continue
        res = cv2.matchTemplate(image_gray, resized_template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        if max_val > best_val:
            best_val = max_val
            best_loc = max_loc
            best_scale = scale
    return best_val, best_loc, best_scale

match_val, top_left, scale = multi_scale_template_match(query_img_color, template)
threshold = 0.5
template_bbox = None
if match_val >= threshold:
    th, tw = template.shape[:2]
    bottom_right = (int(top_left[0] + tw * scale), int(top_left[1] + th * scale))
    template_bbox = [top_left[0], top_left[1], bottom_right[0], bottom_right[1]]
    
    template_img_out = query_img_color.copy()
    cv2.rectangle(template_img_out, top_left, bottom_right, (0, 255, 0), 2)
    cv2.imwrite('template_result.jpg', template_img_out)
    print(f"Template Matching: Logo detected with confidence {match_val:.4f} at scale {scale:.4f}.")
else:
    print(f"Template Matching: No detection above threshold (confidence {match_val:.4f}).")

# --- Compare Results ---
print("\nComparison:")
if orb_bbox and template_bbox:
    iou = compute_iou(orb_bbox, template_bbox)
    print(f"IoU between ORB and Template Matching: {iou:.4f}")
    print(f"ORB: Logo detected with {orb_good_matches_count} good matches.")
    print(f"Template Matching: Logo detected with confidence {match_val:.4f}.")
elif orb_bbox:
    print(f"ORB: Logo detected with {orb_good_matches_count} good matches (Template Matching failed).")
elif template_bbox:
    print(f"Template Matching: Logo detected with confidence {match_val:.4f} (ORB failed).")
else:
    print("Both ORB and Template Matching failed to detect a logo.")