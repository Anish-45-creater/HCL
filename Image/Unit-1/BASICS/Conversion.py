import cv2
import numpy as np
import os


input_image_path = "002.png"  


if not os.path.exists(input_image_path):
    raise FileNotFoundError(f"Error: Image not found at '{input_image_path}'. Please check the file path.")


output_folder = os.path.dirname(input_image_path)
image_bgr = cv2.imread(input_image_path)
if image_bgr is None:
    raise FileNotFoundError(f"Error: Could not load image at '{input_image_path}'. Check file integrity or format.")

# Convert BGR to RGB 
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

# Built-in Methods 
# Convert to grayscale 
gray_image_cv2 = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

# Convert grayscale back to RGB 
rgb_from_gray_cv2 = cv2.cvtColor(gray_image_cv2, cv2.COLOR_GRAY2RGB)

# Manual Conversion Methods
# Manual grayscale conversion 
def manual_grayscale(image):
    weights = np.array([0.299, 0.587, 0.114])  
    gray = np.dot(image[..., :3], weights).astype(np.uint8)
    return gray

gray_image_manual = manual_grayscale(image_rgb)

# Manual conversion from grayscale to RGB 
def manual_gray_to_rgb(gray_image):
    return np.stack([gray_image, gray_image, gray_image], axis=-1)

rgb_from_gray_manual = manual_gray_to_rgb(gray_image_manual)




try:
    original_rgb_path = os.path.join(output_folder, "original_rgb.jpg")
    cv2.imwrite(original_rgb_path, cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))


    gray_cv2_path = os.path.join(output_folder, "gray_cv2.jpg")
    cv2.imwrite(gray_cv2_path, gray_image_cv2)
    
    rgb_cv2_path = os.path.join(output_folder, "rgb_from_gray_cv2.jpg")
    cv2.imwrite(rgb_cv2_path, cv2.cvtColor(rgb_from_gray_cv2, cv2.COLOR_RGB2BGR))
    
    gray_manual_path = os.path.join(output_folder, "gray_manual.jpg")
    cv2.imwrite(gray_manual_path, gray_image_manual)
    
    rgb_manual_path = os.path.join(output_folder, "rgb_from_gray_manual.jpg")
    cv2.imwrite(rgb_manual_path, cv2.cvtColor(rgb_from_gray_manual, cv2.COLOR_RGB2BGR))

except Exception as e:
    print(f"Error saving images: {e}. Check write permissions in '{output_folder}'.")


