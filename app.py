import streamlit as st
import cv2
import numpy as np
import os
import zipfile
from io import BytesIO
from PIL import Image
import streamlit.components.v1 as components
import base64

# --- Core Logic Functions ---

def extract_stickers_logic(img):
    """Core extraction logic for production."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0, 0, 190]) 
    upper_white = np.array([180, 50, 255])
    white_mask = cv2.inRange(hsv, lower_white, upper_white)
    
    kernel_small = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel_small)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_small)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    raw_regions = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 100 and h > 100:
            if h > 800:
                search_range = slice(int(h*0.35), int(h*0.65))
                proj = np.sum(mask[y:y+h, x:x+w], axis=1)
                split_y = np.argmin(proj[search_range]) + int(h*0.35)
                raw_regions.append(((x, y, w, split_y), cnt))
                raw_regions.append(((x, y + split_y, w, h - split_y), cnt))
            else:
                raw_regions.append(((x, y, w, h), cnt))

    raw_regions.sort(key=lambda item: (item[0][1] // 400, item[0][0]))

    results = []
    for i, ((rx, ry, rw, rh), full_cnt) in enumerate(raw_regions, 1):
        pad = 80
        x1, y1 = max(0, rx - pad), max(0, ry - pad)
        x2, y2 = min(img.shape[1], rx + rw + pad), min(img.shape[0], ry + rh + pad)
        crop_img = img[y1:y2, x1:x2]
        h_c, w_c = crop_img.shape[:2]
        
        local_mask = np.zeros((h_c, w_c), dtype=np.uint8)
        shifted_cnt = full_cnt - [x1, y1]
        cv2.drawContours(local_mask, [shifted_cnt], -1, 255, -1)
        
        box_mask = np.zeros_like(local_mask)
        bx, by = rx - x1, ry - y1
        cv2.rectangle(box_mask, (bx, by), (bx + rw, by + rh), 255, -1)
        local_mask = cv2.bitwise_and(local_mask, box_mask)

        final_png = np.zeros((h_c, w_c, 4), dtype=np.uint8)
        shadow_blur = cv2.GaussianBlur(local_mask, (51, 51), 0)
        off_y, off_x = 12, 12 
        M = np.float32([[1, 0, off_x], [0, 1, off_y]])
        shifted_shadow = cv2.warpAffine(shadow_blur, M, (w_c, h_c))
        final_png[:, :, 3] = (shifted_shadow * 0.45).astype(np.uint8)
        
        sticker_rgba = cv2.cvtColor(crop_img, cv2.COLOR_BGR2BGRA)
        sticker_rgba[:, :, 3] = local_mask
        
        alpha_s = sticker_rgba[:, :, 3] / 255.0
        alpha_l = 1.0 - alpha_s
        for c in range(3):
            final_png[:, :, c] = (alpha_s * sticker_rgba[:, :, c] + alpha_l * final_png[:, :, c])
        final_png[:, :, 3] = cv2.addWeighted(final_png[:, :, 3], 1, sticker_rgba[:, :, 3], 1, 0)
        results.append(cv2.cvtColor(final_png, cv2.COLOR_BGRA2RGBA))
    return results

def get_study_steps_logic(img):
    """Generates a dictionary of images representing each step of the process."""
    steps = {}
    steps["1. Original"] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0, 0, 190]) 
    upper_white = np.array([180, 50, 255])
    white_mask = cv2.inRange(hsv, lower_white, upper_white)
    steps["2. HSV White Mask"] = white_mask
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    cleaned_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel)
    cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, kernel)
    steps["3. Morphological Cleaning"] = cleaned_mask
    
    contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    det_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    valid_contours = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 100 and h > 100:
            valid_contours.append(cnt)
            cv2.rectangle(det_img, (x, y), (x + w, y + h), (0, 255, 0), 5)
    steps["4. Object Detection"] = det_img
    
    if valid_contours:
        valid_contours.sort(key=lambda c: (cv2.boundingRect(c)[1], cv2.boundingRect(c)[0]))
        sample_cnt = valid_contours[0]
        x, y, w, h = cv2.boundingRect(sample_cnt)
        pad = 80
        x1, y1 = max(0, x - pad), max(0, y - pad)
        x2, y2 = min(img.shape[1], x + w + pad), min(img.shape[0], y + h + pad)
        
        local_mask = np.zeros((y2 - y1, x2 - x1), dtype=np.uint8)
        shifted_cnt = sample_cnt - [x1, y1]
        cv2.drawContours(local_mask, [shifted_cnt], -1, 255, -1)
        steps["5. Solid Filled Mask (Internal Detail Preserved)"] = local_mask
        
        shadow_blur = cv2.GaussianBlur(local_mask, (51, 51), 0)
        M = np.float32([[1, 0, 12], [0, 1, 12]])
        shadow_shifted = cv2.warpAffine(shadow_blur, M, (x2-x1, y2-y1))
        steps["6. Generated Shadow Layer"] = shadow_shifted
        
        final_png = np.zeros((y2 - y1, x2 - x1, 4), dtype=np.uint8)
        final_png[:, :, 3] = (shadow_shifted * 0.45).astype(np.uint8)
        sticker_rgba = cv2.cvtColor(img[y1:y2, x1:x2], cv2.COLOR_BGR2BGRA)
        sticker_rgba[:, :, 3] = local_mask
        alpha_s = sticker_rgba[:, :, 3] / 255.0
        alpha_l = 1.0 - alpha_s
        for c in range(3):
            final_png[:, :, c] = (alpha_s * sticker_rgba[:, :, c] + alpha_l * final_png[:, :, c])
        final_png[:, :, 3] = cv2.addWeighted(final_png[:, :, 3], 1, sticker_rgba[:, :, 3], 1, 0)
        steps["7. Final Stylized Result"] = cv2.cvtColor(final_png, cv2.COLOR_BGRA2RGBA)

    return steps

def get_image_download_link(img_array, filename):
    """Helper to convert image to base64 for JS download."""
    img_pil = Image.fromarray(img_array)
    buffered = BytesIO()
    img_pil.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/png;base64,{img_str}", filename

# --- Streamlit UI ---
st.set_page_config(page_title="Sticker AI Hub", layout="wide")

# Initialize session state
if 'extracted_stickers' not in st.session_state:
    st.session_state.extracted_stickers = None
if 'study_data' not in st.session_state:
    st.session_state.study_data = None
if 'last_file' not in st.session_state:
    st.session_state.last_file = None

# Sidebar Navigation
st.sidebar.title("🧭 Navigation")
page = st.sidebar.radio("Go to", ["🏠 Main Extractor", "🎓 Study Steps (Tutorial)"])

# --- Page 1: Main Extractor ---
if page == "🏠 Main Extractor":
    st.title("🎨 AI Sticker Extractor")
    st.markdown("Upload your sticker sheet and get transparent PNGs with shadows.")
    
    uploaded_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"], key="main_upload")
    
    if uploaded_file:
        file_id = f"main_{uploaded_file.name}_{uploaded_file.size}"
        if st.session_state.last_file != file_id:
            st.session_state.extracted_stickers = None
            st.session_state.last_file = file_id
            st.rerun()

        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Input Preview", use_container_width=True)

        if st.button("🚀 Run Extraction"):
            with st.spinner("Processing..."):
                st.session_state.extracted_stickers = extract_stickers_logic(image)

    if st.session_state.extracted_stickers:
        st.success(f"Success! {len(st.session_state.extracted_stickers)} items found.")
        
        # Bulk Download Action
        if st.button("📸 Save All to Downloads (Individual Files)", type="primary"):
            js_code = ""
            for i, sticker in enumerate(st.session_state.extracted_stickers):
                data_url, fname = get_image_download_link(sticker, f"sticker_{i+1}.png")
                js_code += f"""
                setTimeout(() => {{
                    const link = document.createElement('a');
                    link.href = '{data_url}';
                    link.download = '{fname}';
                    document.body.appendChild(link);
                    link.click();
                    document.body.removeChild(link);
                }}, {i * 300});
                """
            components.html(f"<script>{js_code}</script>", height=0)
            st.info("Please 'Allow' multiple downloads in your browser.")

        st.markdown("---")
        
        zip_buffer = BytesIO()
        with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zip_file:
            cols = st.columns(4)
            for i, sticker in enumerate(st.session_state.extracted_stickers):
                with cols[i % 4]:
                    st.image(sticker, caption=f"Sticker {i+1}")
                    img_byte_arr = BytesIO()
                    Image.fromarray(sticker).save(img_byte_arr, format='PNG')
                    st.download_button(label=f"💾 Save #{i+1}", data=img_byte_arr.getvalue(), 
                                       file_name=f"sticker_{i+1}.png", mime="image/png", key=f"dl_{i}")
                    zip_file.writestr(f"sticker_{i+1}.png", img_byte_arr.getvalue())
        
        st.download_button("📥 Download All (ZIP Archive)", data=zip_buffer.getvalue(), 
                           file_name="stickers.zip", mime="application/zip", key="dl_zip_main")

# --- Page 2: Study Steps ---
elif page == "🎓 Study Steps (Tutorial)":
    st.title("🎓 How it Works: Step-by-Step")
    st.markdown("Visualize each stage of the Computer Vision algorithm.")
    
    uploaded_file = st.file_uploader("Upload Image to Study", type=["png", "jpg", "jpeg"], key="study_upload")
    
    if uploaded_file:
        file_id = f"study_{uploaded_file.name}_{uploaded_file.size}"
        if st.session_state.last_file != file_id:
            st.session_state.study_data = None
            st.session_state.last_file = file_id
            st.rerun()

        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if st.button("🔍 Visualize Steps"):
            with st.spinner("Analyzing..."):
                st.session_state.study_data = get_study_steps_logic(image)

    if st.session_state.study_data:
        tech_descriptions = {
            "1. Original": """
                **เทคนิค:** Base Image Loading (BGR to RGB)
                **คำอธิบาย:** ขั้นตอนเริ่มต้นคือการโหลดภาพดิจิทัลเข้ามา ใน OpenCV ภาพจะถูกเก็บในรูปแบบ BGR แต่เพื่อแสดงผลบนเว็บเราต้องแปลงเป็น RGB
                ```python
                image = cv2.imread(path)
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                ```
            """,
            "2. HSV White Mask": """
                **เทคนิค:** Color Space Isolation (HSV Thresholding)
                **คำอธิบาย:** เราแปลงภาพเป็นระบบสี HSV เพื่อคัดแยกเฉพาะ 'ขอบขาว' ออกจากพื้นหลังสีชมพู
                ```python
                hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                lower_white = np.array([0, 0, 190]) 
                upper_white = np.array([180, 50, 255])
                white_mask = cv2.inRange(hsv, lower_white, upper_white)
                ```
            """,
            "3. Morphological Cleaning": """
                **เทคนิค:** Morphological Operations (Opening & Closing)
                **คำอธิบาย:** ใช้ Opening เพื่อลบจุดรบกวน (Noise) และ Closing เพื่อเชื่อมรอยโหว่ในเส้นขอบ
                ```python
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
                cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
                cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
                ```
            """,
            "4. Object Detection": """
                **เทคนิค:** Contour Detection & Bounding Box Filtering
                **คำอธิบาย:** ใช้ `findContours` ตรวจจับกลุ่มพิกเซลขาวที่เชื่อมต่อกัน และกรองวัตถุที่ขนาดเล็กเกินไปออก
                ```python
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for cnt in contours:
                    x, y, w, h = cv2.boundingRect(cnt)
                    if w > 100 and h > 100:
                        cv2.rectangle(img, (x, y), (x+w, y+h), (0,255,0), 5)
                ```
            """,
            "5. Solid Filled Mask (Internal Detail Preserved)": """
                **เทคนิค:** Filled Contour Masking (Solid Alpha Channel)
                **คำอธิบาย:** **(สำคัญ)** ถมสีทึบลงในเส้นรอบรูป เพื่อรักษาพิกเซลภายในไม่ให้หายไปตามสีขอบขาว
                ```python
                cv2.drawContours(mask, [contour], -1, 255, thickness=-1)
                ```
            """,
            "6. Generated Shadow Layer": """
                **เทคนิค:** Gaussian Blur & Affine Transformation
                **คำอธิบาย:** ทำเบลอหน้ากากทึบเพื่อให้ขอบเงานุ่มนวล และขยับตำแหน่ง (Offset) ไปทางขวาและล่าง
                ```python
                shadow = cv2.GaussianBlur(mask, (51, 51), 0)
                M = np.float32([[1, 0, 12], [0, 1, 12]])
                shifted = cv2.warpAffine(shadow, M, (w, h))
                ```
            """,
            "7. Final Stylized Result": """
                **เทคนิค:** Alpha Blending & Layer Compositing
                **คำอธิบาย:** รวมเลเยอร์เงาและตัวละครเข้าด้วยกันโดยใช้ค่าความโปร่งใส (Alpha)
                ```python
                sticker_rgba[:, :, 3] = solid_mask
                final[:, :, 3] = shadow_alpha + sticker_alpha
                ```
            """
        }
        for title, img in st.session_state.study_data.items():
            st.subheader(title)
            st.image(img, use_container_width=True)
            if title in tech_descriptions:
                st.info(tech_descriptions[title])
            st.divider()

st.sidebar.markdown("---")
st.sidebar.info("Developed with ❤️ using OpenCV and Streamlit.")
