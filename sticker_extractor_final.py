import cv2
import numpy as np
import os

def extract_stickers_final(image_path, output_dir="result_stickers"):
    """
    [TH] โค้ดสำหรับแยกสติกเกอร์ที่มีขอบขาวออกจากพื้นหลังสี
    
    เทคนิคที่ใช้ (Techniques Used):
    1. HSV Color Isolation: ใช้ระบบสี HSV เพื่อแยกแยะ "สีขาว" ของขอบสติกเกอร์ได้แม่นยำกว่า RGB 
       เพราะแยกค่าความสว่าง (Value) ออกจากสีได้
    2. Morphological Operations: 
       - Opening: กำจัดจุดรบกวน (Noise) เล็กๆ
       - Closing: เชื่อมรอยโหว่เล็กๆ ในขอบให้ต่อเนื่องกัน
    3. Vertical Profiling & Smart Splitting: ตรวจสอบความสูงของภาพ หากสูงเกินไปจะทำการสแกนหา
       จุดที่ "แคบที่สุด" (Valley) เพื่อตัดแยกสติกเกอร์ที่ติดกันออกจากกัน
    4. Solid Filled Contours: แทนที่จะใช้ Mask สีขาวตรงๆ เราหาเส้นรอบรูปแล้ว "ถมดำทึบ" 
       เพื่อให้แน่ใจว่าตัวละครข้างใน (ซึ่งอาจไม่ใช่สีขาว) จะไม่ถูกลบหายไป
    5. Soft Drop Shadow: สร้างเลเยอร์เงาโดยใช้ Gaussian Blur และการขยับตำแหน่ง (Offset)
    6. Alpha Blending: การผสมเลเยอร์โดยใช้ค่าความโปร่งใส เพื่อให้ขอบภาพเรียบเนียน (Anti-aliasing)
    """
    
    # อ่านภาพต้นฉบับ
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image {image_path}")
        return
    
    # --- ขั้นตอนที่ 1: การคัดแยกสี (Color Isolation) ---
    # แปลงเป็น HSV (Hue, Saturation, Value) เพื่อแยกสีขาวออกจากพื้นหลังที่มีสีสัน
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # กำหนดช่วงของสีขาว: Low Saturation (สีไม่สด) และ High Value (สว่างมาก)
    lower_white = np.array([0, 0, 190]) 
    upper_white = np.array([180, 50, 255])
    white_mask = cv2.inRange(hsv, lower_white, upper_white)
    
    # --- ขั้นตอนที่ 2: ปรับแต่งหน้ากาก (Morphological Cleanup) ---
    # MORPH_OPEN: การกร่อนภาพแล้วขยาย (Erode then Dilate) เพื่อกำจัด Noise เล็กๆ
    kernel_small = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel_small)
    # MORPH_CLOSE: การขยายภาพแล้วกร่อน (Dilate then Erode) เพื่อเติมรอยแยกเล็กๆ ในขอบขาว
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_small)
    
    # ค้นหาเส้นรอบรูป (Contours) ของสติกเกอร์แต่ละชิ้น
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    os.makedirs(output_dir, exist_ok=True)
    
    raw_regions = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 100 and h > 100:
            # --- ขั้นตอนที่ 3: การแยกภาพที่ติดกัน (Smart Splitting) ---
            # หากความสูง (h) มากกว่า 800 พิกเซล สันนิษฐานว่ามีสติกเกอร์ 2 ชิ้นติดกันในแนวตั้ง
            if h > 800:
                # สแกนหา "หุบเขา" (Valley) หรือจุดที่มีพิกเซลสีขาวน้อยที่สุดในช่วงกลางภาพ
                search_range = slice(int(h*0.35), int(h*0.65))
                proj = np.sum(mask[y:y+h, x:x+w], axis=1) # รวมจำนวนพิกเซลในแต่ละแถว
                split_y = np.argmin(proj[search_range]) + int(h*0.35)
                
                # แบ่งเป็น 2 ส่วน (บนและล่าง)
                raw_regions.append(((x, y, w, split_y), cnt))
                raw_regions.append(((x, y + split_y, w, h - split_y), cnt))
            else:
                raw_regions.append(((x, y, w, h), cnt))

    # จัดเรียงลำดับภาพจาก บนลงล่าง และ ซ้ายไปขวา
    raw_regions.sort(key=lambda item: (item[0][1] // 400, item[0][0]))

    print(f"Processing {image_path}: Found {len(raw_regions)} stickers...")

    for i, ((rx, ry, rw, rh), full_cnt) in enumerate(raw_regions, 1):
        # เผื่อพื้นที่รอบข้างสำหรับใส่เงาและขอบ (Padding)
        pad = 80
        x1, y1 = max(0, rx - pad), max(0, ry - pad)
        x2, y2 = min(img.shape[1], rx + rw + pad), min(img.shape[0], ry + rh + pad)
        
        crop_img = img[y1:y2, x1:x2]
        h_crop, w_crop = crop_img.shape[:2]
        
        # --- ขั้นตอนที่ 4: การสร้างหน้ากากทึบ (Solid Filled Masking) ---
        # สร้างหน้ากากว่างเปล่าขนาดเท่าภาพที่ Crop
        local_mask = np.zeros((h_crop, w_crop), dtype=np.uint8)
        shifted_cnt = full_cnt - [x1, y1]
        # วาดเส้นรอบรูปและ "ถมสีทึบ" (Thickness = -1) 
        # เพื่อให้ภายในสติกเกอร์ทั้งหมดเป็นสีขาว (พิกเซล 255) ไม่ว่าข้างในจะเป็นสีอะไรก็ตาม
        cv2.drawContours(local_mask, [shifted_cnt], -1, 255, -1)
        
        # ตัดแบ่งหน้ากากหากเป็นกรณีที่ภาพติดกัน
        box_mask = np.zeros_like(local_mask)
        bx, by = rx - x1, ry - y1
        cv2.rectangle(box_mask, (bx, by), (bx + rw, by + rh), 255, -1)
        local_mask = cv2.bitwise_and(local_mask, box_mask)

        # --- ขั้นตอนที่ 5: การสร้างเงาและรวมภาพ (Shadow & Compositing) ---
        final_png = np.zeros((h_crop, w_crop, 4), dtype=np.uint8)
        
        # สร้างเงา (Drop Shadow):
        # 1. ทำเบลอหน้ากากทึบเพื่อให้ขอบฟุ้ง
        shadow_blur = cv2.GaussianBlur(local_mask, (51, 51), 0)
        # 2. ขยับตำแหน่งเงาลงด้านล่างขวา 12 พิกเซล
        off_y, off_x = 12, 12 
        M = np.float32([[1, 0, off_x], [0, 1, off_y]])
        shifted_shadow = cv2.warpAffine(shadow_blur, M, (w_crop, h_crop))
        # 3. กำหนดความโปร่งใสของเงา (45%) ใน Alpha Channel
        final_png[:, :, 3] = (shifted_shadow * 0.45).astype(np.uint8)
        
        # เตรียมเลเยอร์สติกเกอร์
        sticker_rgba = cv2.cvtColor(crop_img, cv2.COLOR_BGR2BGRA)
        sticker_rgba[:, :, 3] = local_mask # ใช้หน้ากากทึบเป็นตัวกำหนดความโปร่งใส
        
        # ผสมเลเยอร์ (Alpha Blending): เพื่อความเนียนของขอบระหว่างเงากับตัวละคร
        alpha_s = sticker_rgba[:, :, 3] / 255.0
        alpha_l = 1.0 - alpha_s
        for c in range(3):
            final_png[:, :, c] = (alpha_s * sticker_rgba[:, :, c] + alpha_l * final_png[:, :, c])
        final_png[:, :, 3] = cv2.addWeighted(final_png[:, :, 3], 1, sticker_rgba[:, :, 3], 1, 0)

        # บันทึกไฟล์ PNG
        out_name = f"{base_name}_{i}.png"
        cv2.imwrite(os.path.join(output_dir, out_name), final_png)

if __name__ == "__main__":
    target_dir = "."
    output_dir = "result_stickers"
    
    # ค้นหาภาพใน Directory
    images = [f for f in os.listdir(target_dir) if f.endswith(".png") and 
              ("Generated" in f or "แซน" in f) and 
              "debug" not in f and "result" not in f]
    
    if not images:
        print("No target images found.")
    else:
        for img_file in images:
            extract_stickers_final(img_file, output_dir)
        print(f"\nCompleted! Results saved in '{output_dir}'.")
