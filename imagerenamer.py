import streamlit as st
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch
import io
import zipfile
import os
import re

# Load model WITHOUT `.to(device)` to avoid NotImplementedError
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Stopwords to clean from captions
STOPWORDS = {"a", "an", "the", "with", "and", "on", "in", "of", "at", "to", "by", "for", "from"}

# Streamlit UI
st.set_page_config(page_title="AI Image Renamer")
st.title("🧠 AI-Powered T-shirt Graphic Renamer")
st.write("Upload your .png, .jpg, or .jpeg T-shirt designs. The AI will generate names and give you a zip file.")

uploaded_files = st.file_uploader("Upload image files", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

if uploaded_files and st.button("Generate & Download ZIP"):
    zip_buffer = io.BytesIO()
    total_files = len(uploaded_files)
    progress_bar = st.progress(0, text="Starting...")

    with zipfile.ZipFile(zip_buffer, "w") as zip_file:
        for i, uploaded_file in enumerate(uploaded_files):
            # Load original image and keep it intact
            original_image = Image.open(uploaded_file)

            # Resize COPY for AI captioning
            resized_image = original_image.copy()
            resized_image = resized_image.convert("RGB")
            resized_image.thumbnail((512, 512))

            # Caption generation
            inputs = processor(images=resized_image, return_tensors="pt")
            with torch.no_grad():
                output = model.generate(**inputs, max_length=20)
            raw_caption = processor.decode(output[0], skip_special_tokens=True)

            # Clean caption
            filtered_words = [
                word for word in re.findall(r"\w+", raw_caption.lower())
                if word not in STOPWORDS
            ]
            clean_caption = "_".join(filtered_words) or "image"

            ext = os.path.splitext(uploaded_file.name)[1]
            new_filename = f"{clean_caption}{ext}"

            # Save original image (preserving transparency)
            image_bytes = io.BytesIO()
            original_image.save(image_bytes, format=original_image.format or "PNG")
            zip_file.writestr(new_filename, image_bytes.getvalue())

            # Update progress bar
            progress = (i + 1) / total_files
            progress_bar.progress(progress, text=f"Processing {i + 1} of {total_files} images...")

    progress_bar.empty()
    zip_buffer.seek(0)

    st.success("✅ All files processed successfully!")
    st.download_button(
        label="📥 Download Renamed Images ZIP",
        data=zip_buffer,
        file_name="renamed_images.zip",
        mime="application/zip"
    )
