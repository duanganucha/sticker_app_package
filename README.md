# 🎨 AI Sticker Extractor

A Streamlit web application to extract stickers with white borders from image sheets using OpenCV.

## 🌟 Features
- **Automatic Extraction**: Detects stickers based on their white borders.
- **Smart Splitting**: Automatically separates stickers that are touching or joined.
- **Solid Masks**: Ensures character details aren't lost during extraction.
- **Stylized Effects**: Adds customizable white borders and soft drop shadows.
- **Study Mode**: Educational page to see the step-by-step image processing logic.

## 🛠️ Installation
1. Clone this repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 Running Locally
```bash
streamlit run app.py
```

## ☁️ Deployment
This project is ready to be deployed on **Streamlit Community Cloud** or **Hugging Face Spaces**. 
Ensure `requirements.txt` is present in the root directory of your deployment.
