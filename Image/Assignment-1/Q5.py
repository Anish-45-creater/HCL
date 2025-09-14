import cv2
import numpy as np
import os

# Define HSV ranges for ripe and unripe colors
# Green (unripe): Hue 35-85, Sat 100-255, Val 100-255
unripe_lower = np.array([35, 100, 100])
unripe_upper = np.array([85, 255, 255])

# Yellow (ripe banana): Hue 20-35, Sat 100-255, Val 100-255
yellow_lower = np.array([20, 100, 100])
yellow_upper = np.array([35, 255, 255])

# Red (ripe apple): Hue 0-10 or 160-180 (for red), Sat 100-255, Val 100-255
red_lower1 = np.array([0, 100, 100])
red_upper1 = np.array([10, 255, 255])
red_lower2 = np.array([160, 100, 100])
red_upper2 = np.array([180, 255, 255])

# Function to classify image as ripe or unripe
def classify_fruit(image_path, fruit_type):
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error loading {image_path}")
        return "Unknown"
    
    # Convert to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Create masks
    unripe_mask = cv2.inRange(hsv, unripe_lower, unripe_upper)
    
    if fruit_type == 'banana':
        ripe_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
    elif fruit_type == 'apple':
        ripe_mask1 = cv2.inRange(hsv, red_lower1, red_upper1)
        ripe_mask2 = cv2.inRange(hsv, red_lower2, red_upper2)
        ripe_mask = cv2.bitwise_or(ripe_mask1, ripe_mask2)
    else:
        return "Unknown fruit type"
    
    # Compute area (number of pixels)
    unripe_area = np.sum(unripe_mask > 0)
    ripe_area = np.sum(ripe_mask > 0)
    
    # Classify based on which area is larger (assuming fruit dominates the image)
    if ripe_area > unripe_area:
        return "Ripe"
    else:
        return "Unripe"

# Dataset directory
data_dir = 'Data'

# Sample images (at least 5: 2 ripe apple, 1 unripe apple, 1 ripe banana, 1 unripe banana)
samples = [
    {'path': os.path.join(data_dir, 'ripe apple', '1.jpg'), 'fruit': 'apple', 'expected': 'Ripe'},
    {'path': os.path.join(data_dir, 'ripe apple', '2.jpg'), 'fruit': 'apple', 'expected': 'Ripe'},
    {'path': os.path.join(data_dir, 'unripe apple', '1.jpg'), 'fruit': 'apple', 'expected': 'Unripe'},
    {'path': os.path.join(data_dir, 'ripe banana', '1.jpg'), 'fruit': 'banana', 'expected': 'Ripe'},
    {'path': os.path.join(data_dir, 'unripe banana', '1.jpg'), 'fruit': 'banana', 'expected': 'Unripe'}
]

# Test on samples
for sample in samples:
    classification = classify_fruit(sample['path'], sample['fruit'])
    print(f"Image: {sample['path']}, Expected: {sample['expected']}, Classified: {classification}")