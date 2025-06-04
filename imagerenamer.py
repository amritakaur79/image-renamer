import streamlit as st
from PIL import Image
import os
import io
import re
import torch
import zipfile
from transformers import BlipProcessor, BlipForConditionalGeneration

# --- PAGE CONFIG ---
st.set_page_config(page_title="AI T-Shirt Graphic Renamer", layout="centered")
st.title("🧠 AI T-Shirt Graphic Renamer")

st.markdown("""
Upload your **.png, .jpg, or .jpeg** T-shirt graphics below.  
Then click **Generate** to let the AI rename your graphics and download a ZIP.
""")

# --- LOAD MODEL ---
@st.cache_resource
def load_blip():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

processor, model = load_blip()

# --- CAPTION CLEANER ---
STOPWORDS = {"a", "an", "the", "with", "and", "on", "in", "of", "at", "to", "by", "for", "from"}

def clean_caption(caption, existing_names):
    words = re.findall(r'\w+', caption.lower())
    filtered = [w for w in words if w not in STOPWORDS]
    base_name = "_".join(filtered) or "graphic"
    
    final_name = base_name
    i = 1
    while final_name in existing_names:
        final_name = f"{base_name}_{i}"
        i += 1
    existing_names.add(final_name)
    return final_name

# --- GENERATE CAPTION ---
def generate_caption(image):
    image = image.convert("RGB")
    inputs = processor(image, return_tensors="pt")
    out = model.generate(**inputs)
    return processor.decode(out[0], skip_special_tokens=True)

# --- FILE UPLOADER ---
uploaded_files = st.file_uploader(
    "Upload your images here", 
    type=["png", "jpg", "jpeg"], 
    accept_multiple_files=True
)

# --- GENERATE BUTTON ---
if uploaded_files:
    if st.button("🚀 Generate Renamed Files"):
        existing_names = set()
        zip_buffer = io.BytesIO()
        renamed_files = []

        with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED) as zipf:
            for file in uploaded_files:
                st.write(f"Processing: `{file.name}`")
                try:
                    img = Image.open(file)
                    caption = generate_caption(img)
                    new_filename = clean_caption(caption, existing_names)
                    ext = os.path.splitext(file.name)[-1]
                    final_filename = f"{new_filename}{ext}"

                    img_byte_arr = io.BytesIO()
                    img.save(img_byte_arr, format=img.format)
                    zipf.writestr(final_filename, img_byte_arr.getvalue())
                    renamed_files.append((file.name, final_filename))
                except Exception as e:
                    st.error(f"Failed to process `{file.name}`: {e}")

        st.success("✅ Images renamed successfully!")
        st.download_button(
            label="📦 Download Renamed Images (ZIP)",
            data=zip_buffer.getvalue(),
            file_name="renamed_graphics.zip",
            mime="application/zip"
        )

        st.subheader("📝 Preview:")
        for old_name, new_name in renamed_files:
            st.write(f"🖼️ `{old_name}` ➡️ `{new_name}`")
