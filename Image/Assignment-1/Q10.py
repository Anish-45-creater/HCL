import os
import numpy as np
from scipy.ndimage import gaussian_filter, sobel, binary_dilation, binary_erosion
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Manual Canny edge detection implementation
def canny_edge_detection(image, sigma=0.5, low_threshold=0.05, high_threshold=0.15):
    smoothed = gaussian_filter(image, sigma=sigma)
    sx = sobel(smoothed, axis=0, mode='constant')
    sy = sobel(smoothed, axis=1, mode='constant')
    magnitude = np.hypot(sx, sy)
    magnitude = magnitude / magnitude.max()
    
    angle = np.arctan2(sy, sx)
    angle_quantized = np.round(angle / (np.pi/4)) % 4
    suppressed = np.zeros_like(magnitude)
    for i in range(1, magnitude.shape[0]-1):
        for j in range(1, magnitude.shape[1]-1):
            if magnitude[i,j] > 0:
                if angle_quantized[i,j] == 0 and magnitude[i,j] >= max(magnitude[i,j-1], magnitude[i,j+1]):
                    suppressed[i,j] = magnitude[i,j]
                elif angle_quantized[i,j] == 1 and magnitude[i,j] >= max(magnitude[i-1,j+1], magnitude[i+1,j-1]):
                    suppressed[i,j] = magnitude[i,j]
                elif angle_quantized[i,j] == 2 and magnitude[i,j] >= max(magnitude[i-1,j], magnitude[i+1,j]):
                    suppressed[i,j] = magnitude[i,j]
                elif angle_quantized[i,j] == 3 and magnitude[i,j] >= max(magnitude[i-1,j-1], magnitude[i+1,j+1]):
                    suppressed[i,j] = magnitude[i,j]
    
    edges = np.zeros_like(suppressed)
    high_mask = suppressed > high_threshold
    low_mask = (suppressed >= low_threshold) & (suppressed <= high_threshold)
    edges[high_mask] = 1
    connected = binary_dilation(high_mask)
    edges[low_mask & connected] = 1
    
    return edges

# Morphology to detect defects (broken tracks or disjoints)
def detect_defects(edges):
    thinned = binary_erosion(edges, iterations=1)
    closed = binary_dilation(thinned, iterations=2)
    
    from scipy.ndimage import label
    labeled, num_features = label(1 - closed)  # Label gaps/discontinuities
    defects = []
    for i in range(1, num_features + 1):
        component = (labeled == i)
        if np.sum(component) > 20:  
            y, x = np.where(component)
            defects.append((min(x), min(y), max(x) - min(x), max(y) - min(y)))
    return defects

def process_and_display_image(filepath):
    image = plt.imread(filepath)
    if len(image.shape) == 3:
        gray = np.dot(image[..., :3], [0.299, 0.587, 0.114])
    else:
        gray = image
    
    edges = canny_edge_detection(gray)
    defects = detect_defects(edges)
    
    fig, ax = plt.subplots()
    ax.imshow(image)
    for (x, y, w, h) in defects:
        rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        ax.text(x, y - 10, 'Broken Track/Disjoint', color='r')
    
    plt.title('Regions with Broken Tracks or Disjoints')
    plt.show()

base_path = 'SolDef_AI\\Dataset\\CS7\\TR_TH\\V2'  
image_filename = 'WIN_20220822_15_44_38_Pro.jpg'
image_path = os.path.join(base_path, image_filename)

if not os.path.exists(image_path):
    print(f"Image not found at {image_path}. Please ensure the script is run from the correct directory (e.g., C:\\Users\\ANISH KARTHIK\\Desktop\\python\\HCL\\ML\\HandsOn\\) and the image is in 'SolDef_AI\\Dataset\\CS7\\TR_TH\\V2'.")
else:
    process_and_display_image(image_path)