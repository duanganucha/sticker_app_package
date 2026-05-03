import cv2
import numpy as np
import os

def extract_stickers_final(image_path, output_dir="result_stickers"):
    """
    Extracts stickers from a sheet with white borders on a colored background.
    - Preserves solid interiors (no hollow characters).
    - Automatically splits joined stickers.
    - Adds white border and soft drop shadow.
    - Saves as transparent PNGs.
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image {image_path}")
        return
    
    # 1. Isolate the white borders using HSV color space
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Range for white (adjust if background is very light)
    lower_white = np.array([0, 0, 190]) 
    upper_white = np.array([180, 50, 255])
    white_mask = cv2.inRange(hsv, lower_white, upper_white)
    
    # 2. Pre-process mask to find sticker regions
    # Opening breaks thin bridges between separate stickers
    kernel_small = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel_small)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_small)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    os.makedirs(output_dir, exist_ok=True)
    
    raw_regions = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        # Filter out small noise
        if w > 100 and h > 100:
            # Smart Splitting: If region is too tall, it's likely 2 stickers combined
            if h > 800:
                search_range = slice(int(h*0.35), int(h*0.65))
                proj = np.sum(mask[y:y+h, x:x+w], axis=1)
                split_y = np.argmin(proj[search_range]) + int(h*0.35)
                
                # Split into Top and Bottom regions
                raw_regions.append(((x, y, w, split_y), cnt))
                raw_regions.append(((x, y + split_y, w, h - split_y), cnt))
            else:
                raw_regions.append(((x, y, w, h), cnt))

    # Sort row-major (Top-to-bottom, Left-to-right) for consistent numbering
    raw_regions.sort(key=lambda item: (item[0][1] // 400, item[0][0]))

    print(f"Processing {image_path}: Found {len(raw_regions)} stickers...")

    for i, ((rx, ry, rw, rh), full_cnt) in enumerate(raw_regions, 1):
        # Padding for shadow and edges
        pad = 80
        x1, y1 = max(0, rx - pad), max(0, ry - pad)
        x2, y2 = min(img.shape[1], rx + rw + pad), min(img.shape[0], ry + rh + pad)
        
        crop_img = img[y1:y2, x1:x2]
        h_crop, w_crop = crop_img.shape[:2]
        
        # 3. Create SOLID mask for the sticker interior
        local_mask = np.zeros((h_crop, w_crop), dtype=np.uint8)
        shifted_cnt = full_cnt - [x1, y1]
        # Fill the entire contour to prevent hollow centers
        cv2.drawContours(local_mask, [shifted_cnt], -1, 255, -1)
        
        # Intersect with the bounding box to handle the split if necessary
        box_mask = np.zeros_like(local_mask)
        bx, by = rx - x1, ry - y1
        cv2.rectangle(box_mask, (bx, by), (bx + rw, by + rh), 255, -1)
        local_mask = cv2.bitwise_and(local_mask, box_mask)

        # 4. Build final PNG with shadow effect
        final_png = np.zeros((h_crop, w_crop, 4), dtype=np.uint8)
        
        # Generate Drop Shadow
        shadow_blur = cv2.GaussianBlur(local_mask, (51, 51), 0)
        off_y, off_x = 12, 12 # Shadow offset
        M = np.float32([[1, 0, off_x], [0, 1, off_y]])
        shifted_shadow = cv2.warpAffine(shadow_blur, M, (w_crop, h_crop))
        final_png[:, :, 3] = (shifted_shadow * 0.45).astype(np.uint8) # 45% Opacity
        
        # Character/Sticker Layer
        sticker_rgba = cv2.cvtColor(crop_img, cv2.COLOR_BGR2BGRA)
        sticker_rgba[:, :, 3] = local_mask
        
        # Alpha blending for smooth compositing
        alpha_s = sticker_rgba[:, :, 3] / 255.0
        alpha_l = 1.0 - alpha_s
        for c in range(3):
            final_png[:, :, c] = (alpha_s * sticker_rgba[:, :, c] + alpha_l * final_png[:, :, c])
        final_png[:, :, 3] = cv2.addWeighted(final_png[:, :, 3], 1, sticker_rgba[:, :, 3], 1, 0)

        # Save result
        out_name = f"{base_name}_{i}.png"
        cv2.imwrite(os.path.join(output_dir, out_name), final_png)

if __name__ == "__main__":
    target_dir = "."
    output_dir = "result_stickers"
    
    # Process all images starting with 'แซน' or 'Gemini'
    images = [f for f in os.listdir(target_dir) if f.endswith(".png") and 
              ("Generated" in f or "แซน" in f) and 
              "debug" not in f and "result" not in f]
    
    if not images:
        print("No target images found in the directory.")
    else:
        for img_file in images:
            extract_stickers_final(img_file, output_dir)
        print(f"\nCompleted! Check the '{output_dir}' folder for results.")
