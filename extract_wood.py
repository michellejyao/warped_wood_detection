import torch
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
from PIL import Image
import numpy as np
import cv2
import time
import warnings
from constants import (
    RGB_IMAGE_PATH, WOOD_REFERENCE_PATH, WOOD_PANEL_MASK_PATH,
    WOOD_PANEL_DEPTH_PATH, DEPTH_MAP_PATH, CLIPSEG_MODEL, SEGMENTATION_THRESHOLD,
    TEXT_OR_IMAGE, TEXT_PROMPT
)

warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
start_time = time.time()

print("=" * 50)
print("Wood Panel Segmentation with CLIPSeg")
print("=" * 50)

# 1) Load RGB
print("\n1. Loading RGB image...")
try:
    rgb_image = Image.open(RGB_IMAGE_PATH).convert("RGB")
    print(f"✓ Loaded RGB image: {rgb_image.size}")
except Exception as e:
    print(f"✗ Error loading RGB image: {e}")
    exit(1)

# 2) Load CLIPSeg
print("\n2. Loading CLIPSeg model...")
try:
    processor = CLIPSegProcessor.from_pretrained(CLIPSEG_MODEL)
    model = CLIPSegForImageSegmentation.from_pretrained(CLIPSEG_MODEL)
    model.eval()
    print("✓ CLIPSeg model loaded successfully")
except Exception as e:
    print(f"✗ Error loading CLIPSeg model: {e}")
    exit(1)

# 3) Choose prompt
use_text_prompt = TEXT_OR_IMAGE
text_prompt = TEXT_PROMPT
print(f"\n3. Running segmentation...")
if use_text_prompt:
    print(f"Using text prompt: '{text_prompt}'")
else:
    print("Using reference image for segmentation")
    try:
        reference_image = Image.open(WOOD_REFERENCE_PATH).convert("RGB")
        print(f"✓ Loaded reference image: {reference_image.size}")
    except Exception as e:
        print(f"✗ Error loading reference image: {e}")
        exit(1)

# 4) Prepare inputs
try:
    if use_text_prompt:
        inputs = processor(text=[text_prompt], images=[rgb_image], return_tensors="pt")
    else:
        encoded_image = processor(images=[rgb_image], return_tensors="pt")
        encoded_prompt = processor(images=[reference_image], return_tensors="pt")
    print("✓ Inputs prepared for CLIPSeg")
except Exception as e:
    print(f"✗ Error preparing inputs: {e}")
    exit(1)

# 5) Run model
try:
    with torch.no_grad():
        if use_text_prompt:
            outputs = model(**inputs)
        else:
            outputs = model(**encoded_image, conditional_pixel_values=encoded_prompt.pixel_values)
        mask_logits = outputs.logits[0]  # (352,352)
    print("✓ Segmentation completed")
except Exception as e:
    print(f"✗ Error during segmentation: {e}")
    exit(1)

# 6) Post-process mask: resize (linear) -> threshold -> morphology
print("\n4. Processing segmentation mask...")
mask_prob = torch.sigmoid(mask_logits).cpu().numpy().astype(np.float32)
mask_resized = cv2.resize(mask_prob, rgb_image.size, interpolation=cv2.INTER_LINEAR)

# Save soft mask for debugging/tuning (float32 .npy)
np.save("soft_mask.npy", mask_resized)

# Threshold to binary 0/255
mask_binary = (mask_resized > SEGMENTATION_THRESHOLD).astype(np.uint8) * 255
print(f"✓ Applied threshold: {SEGMENTATION_THRESHOLD}")

# Morphological refine: open (denoise) -> close (fill holes) -> light dilate (recover edges)
kernel = np.ones((3, 3), np.uint8)
mask_refined = cv2.morphologyEx(mask_binary, cv2.MORPH_OPEN, kernel, iterations=1)
mask_refined = cv2.morphologyEx(mask_refined, cv2.MORPH_CLOSE, kernel, iterations=1)
mask_refined = cv2.dilate(mask_refined, kernel, iterations=1)

# Save mask
cv2.imwrite(WOOD_PANEL_MASK_PATH, mask_refined)
print(f"✓ Segmentation mask saved: {WOOD_PANEL_MASK_PATH}")

# 7) Load depth (must be 16-bit from image_output)
print("\n5. Loading depth map...")
try:
    depth_map = cv2.imread(DEPTH_MAP_PATH, cv2.IMREAD_UNCHANGED)
    if depth_map is None:
        raise FileNotFoundError(f"Could not load depth map from {DEPTH_MAP_PATH}")
    print(f"✓ Loaded depth map: shape={depth_map.shape}, dtype={depth_map.dtype}")
except Exception as e:
    print(f"✗ Error loading depth map: {e}")
    exit(1)

# 8) Ensure sizes match; resize mask to depth (nearest)
print("\n6. Applying mask to depth map...")
if depth_map.shape[:2] != mask_refined.shape:
    mask_resized_to_depth = cv2.resize(
        mask_refined, (depth_map.shape[1], depth_map.shape[0]), interpolation=cv2.INTER_NEAREST
    )
    print(f"✓ Resized mask {mask_refined.shape} -> {mask_resized_to_depth.shape}")
else:
    mask_resized_to_depth = mask_refined
    print("✓ Mask and depth map sizes match")

# Apply mask to 16-bit depth (keep where 255, else 0)
wood_panel_depth = np.where(mask_resized_to_depth == 255, depth_map, 0).astype(depth_map.dtype)

# Coverage KPI
wood_pixels = np.count_nonzero(mask_resized_to_depth == 255)
total_pixels = mask_resized_to_depth.size
percentage = (wood_pixels / total_pixels) * 100.0
print(f"✓ Wood panel coverage: {wood_pixels}/{total_pixels} pixels ({percentage:.1f}%)")

# Optional guard: skip if mask clearly failed/flooded
MIN_COVER = 1.0   # %
MAX_COVER = 95.0  # %
if not (MIN_COVER <= percentage <= MAX_COVER):
    print("⚠️ Coverage out of expected range; consider adjusting SEGMENTATION_THRESHOLD.")

# 9) Save masked depth (still 16-bit)
cv2.imwrite(WOOD_PANEL_DEPTH_PATH, wood_panel_depth)
print(f"✓ Masked depth map saved: {WOOD_PANEL_DEPTH_PATH}")

print("\n Segmentation completed successfully!")
print(f"Generated files:\n  - {WOOD_PANEL_MASK_PATH}\n  - {WOOD_PANEL_DEPTH_PATH}")
print(f"Execution time: {time.time() - start_time:.2f} seconds")
