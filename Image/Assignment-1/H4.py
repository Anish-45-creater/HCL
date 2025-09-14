import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Base directory for MVTec AD dataset (screw category)
base_dir = './archive/screw/'

# Image paths: good image and defective image
img1_path = os.path.join(base_dir, 'train', 'good', '000.png')  # Reference image
img2_path = os.path.join(base_dir, 'test', 'scratch_head', '000.png')  # Test image with defect

# Load images
img1 = cv2.imread(img1_path)
img2 = cv2.imread(img2_path)
if img1 is None or img2 is None:
    raise ValueError(f"Images not found at {img1_path} or {img2_path}")

img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# SIFT Feature Extraction
sift = cv2.SIFT_create()
kp1_sift, des1_sift = sift.detectAndCompute(img1_gray, None)
kp2_sift, des2_sift = sift.detectAndCompute(img2_gray, None)



# Debug: Print number of keypoints
print(f"SIFT keypoints: img1={len(kp1_sift)}, img2={len(kp2_sift)}")


# Feature Matching for SIFT using FLANN
if des1_sift is not None and des2_sift is not None:
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches_sift = flann.knnMatch(des1_sift, des2_sift, k=2)

    good_matches_sift = []
    for m, n in matches_sift:
        if m.distance < 0.7 * n.distance:
            good_matches_sift.append(m)

    match_img_sift = cv2.drawMatches(img1, kp1_sift, img2, kp2_sift, good_matches_sift, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
else:
    print("No SIFT descriptors found.")
    match_img_sift = np.zeros_like(img1)


# Visualization
plt.figure(figsize=(12, 6))

plt.subplot(1, 1, 1)
plt.imshow(cv2.cvtColor(match_img_sift, cv2.COLOR_BGR2RGB))
plt.title(f'SIFT Matches (Good: {len(good_matches_sift)})')
plt.axis('off')



plt.tight_layout()
plt.savefig('feature_matches_comparison.png')
plt.show()

# Save the matched images
cv2.imwrite('sift_matches.jpg', match_img_sift)
print("Matched images saved as 'sift_matches.jpg")