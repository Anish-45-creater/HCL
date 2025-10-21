import cv2
import numpy as np

# ==============================
# 1. Define parameters and helper
# ==============================
def measure_pixel_resolution(img_path, known_length_mm, pixels_across):
    """Estimate pixel resolution (mm/pixel) for a given image."""
    pixel_res = known_length_mm / pixels_across
    print(f"{img_path}: Pixel resolution = {pixel_res:.3f} mm/pixel")
    return pixel_res

def compute_dof(focal_length_mm, f_number, subject_distance_mm, coc_mm):
    """Compute depth of field (DOF) using optical formula."""
    dof_near = (subject_distance_mm * (subject_distance_mm - focal_length_mm)) / \
               (subject_distance_mm + (focal_length_mm / f_number) * (subject_distance_mm / coc_mm) - focal_length_mm)
    dof_far = (subject_distance_mm * (subject_distance_mm - focal_length_mm)) / \
              (subject_distance_mm - (focal_length_mm / f_number) * (subject_distance_mm / coc_mm) - focal_length_mm)
    dof = dof_far - dof_near if np.isfinite(dof_far) else np.inf
    return dof, dof_near, dof_far

# ==============================
# 2. Inputs (replace with actual measured pixel values)
# ==============================
images_info = [
    {"file": "35mm.png", "focal_length": 35, "pixels_across": 60},
    {"file": "50mm.png", "focal_length": 50, "pixels_across": 50},
    {"file": "85mm.png", "focal_length": 85, "pixels_across": 45}
]

known_length_mm = 100   # Known real-world object size
f_number = 5.6          # Aperture (can change)
subject_distance_mm = 1000  # Distance to subject

# ==============================
# 3. Process each image
# ==============================
results = []

for info in images_info:
    img_path = info["file"]
    focal_length = info["focal_length"]
    pixels_across = info["pixels_across"]

    # Load image (optional visualization)
    try:
        img = cv2.imread(img_path)
        if img is not None:
            print(f"Loaded {img_path}, size: {img.shape[1]}x{img.shape[0]}")
        else:
            print(f"Warning: Could not load {img_path}, using default resolution assumption.")
    except Exception as e:
        print(f"Error loading {img_path}: {e}")

    # Compute pixel resolution
    pixel_res = measure_pixel_resolution(img_path, known_length_mm, pixels_across)

    # Approximate CoC (circle of confusion)
    coc_mm = 2 * pixel_res

    # Compute DOF
    dof, dof_near, dof_far = compute_dof(focal_length, f_number, subject_distance_mm, coc_mm)

    results.append({
        "Image": img_path,
        "Focal Length (mm)": focal_length,
        "CoC (mm)": coc_mm,
        "DOF Near (mm)": dof_near,
        "DOF Far (mm)": dof_far,
        "Total DOF (mm)": dof
    })

# ==============================
# 4. Compare results
# ==============================
import pandas as pd
df = pd.DataFrame(results)
print("\n=== Depth of Field Comparison ===")
print(df.to_string(index=False))
