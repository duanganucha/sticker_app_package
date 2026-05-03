import cv2
import numpy as np
import os

def create_educational_steps(image_path, output_dir="study_steps"):
    """
    Saves intermediate images for each step of the sticker extraction process
    to help study and understand the algorithm.
    """
    img = cv2.imread(image_path)
    if img is None: return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # --- STEP 1: Original Image ---
    cv2.imwrite(os.path.join(output_dir, "step1_original.png"), img)
    
    # --- STEP 2: HSV White Masking ---
    # Isolating the white borders
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0, 0, 190]) 
    upper_white = np.array([180, 50, 255])
    white_mask = cv2.inRange(hsv, lower_white, upper_white)
    cv2.imwrite(os.path.join(output_dir, "step2_white_mask.png"), white_mask)
    
    # --- STEP 3: Cleaned Mask (Opening/Closing) ---
    # Breaking thin bridges and smoothing edges
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    cleaned_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel)
    cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, kernel)
    cv2.imwrite(os.path.join(output_dir, "step3_cleaned_mask.png"), cleaned_mask)
    
    # --- STEP 4: Finding Contours & Detecting Objects ---
    # Drawing bounding boxes on the original image to show detection
    contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detection_img = img.copy()
    valid_contours = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 100 and h > 100:
            valid_contours.append(cnt)
            cv2.rectangle(detection_img, (x, y), (x + w, y + h), (0, 255, 0), 3)
    cv2.imwrite(os.path.join(output_dir, "step4_detections.png"), detection_img)
    
    # Pick one sample sticker (e.g., the first one) for detailed steps
    if not valid_contours: return
    
    # Sort them to get the first one (top-left)
    valid_contours.sort(key=lambda c: (cv2.boundingRect(c)[1], cv2.boundingRect(c)[0]))
    sample_cnt = valid_contours[0]
    x, y, w, h = cv2.boundingRect(sample_cnt)
    
    # --- STEP 5: Solid Filled Mask (The secret to fixing hollow centers) ---
    pad = 80
    x1, y1 = max(0, x - pad), max(0, y - pad)
    x2, y2 = min(img.shape[1], x + w + pad), min(img.shape[0], y + h + pad)
    local_crop = img[y1:y2, x1:x2]
    
    solid_mask = np.zeros((y2 - y1, x2 - x1), dtype=np.uint8)
    shifted_cnt = sample_cnt - [x1, y1]
    cv2.drawContours(solid_mask, [shifted_cnt], -1, 255, -1) # -1 means FILL
    cv2.imwrite(os.path.join(output_dir, "step5_solid_filled_mask.png"), solid_mask)
    
    # --- STEP 6: Shadow Generation ---
    # Blur the solid mask to create the shadow shape
    shadow_blur = cv2.GaussianBlur(solid_mask, (51, 51), 0)
    # Shift the shadow
    off_y, off_x = 12, 12
    M = np.float32([[1, 0, off_x], [0, 1, off_y]])
    shadow_shifted = cv2.warpAffine(shadow_blur, M, (x2-x1, y2-y1))
    
    # Save shadow layer as a visible grayscale image
    cv2.imwrite(os.path.join(output_dir, "step6_shadow_layer.png"), shadow_shifted)
    
    # --- STEP 7: Final Compositing (Alpha Blending) ---
    final_png = np.zeros((y2 - y1, x2 - x1, 4), dtype=np.uint8)
    # Set Shadow alpha
    final_png[:, :, 3] = (shadow_shifted * 0.45).astype(np.uint8)
    # Add Character (RGBA)
    sticker_rgba = cv2.cvtColor(local_crop, cv2.COLOR_BGR2BGRA)
    sticker_rgba[:, :, 3] = solid_mask
    
    # Blend layers
    alpha_s = sticker_rgba[:, :, 3] / 255.0
    alpha_l = 1.0 - alpha_s
    for c in range(3):
        final_png[:, :, c] = (alpha_s * sticker_rgba[:, :, c] + alpha_l * final_png[:, :, c])
    final_png[:, :, 3] = cv2.addWeighted(final_png[:, :, 3], 1, sticker_rgba[:, :, 3], 1, 0)
    
    cv2.imwrite(os.path.join(output_dir, "step7_final_composite.png"), final_png)

if __name__ == "__main__":
    # Choose a sample image to study
    sample = "แซน_Generated_Image_b8ah3bb8ah3bb8ah.png"
    if os.path.exists(sample):
        print(f"Creating educational step-by-step images for {sample}...")
        create_educational_steps(sample)
        print("Done! Open the 'study_steps' folder to see each stage.")
    else:
        print("Sample image not found for study.")
