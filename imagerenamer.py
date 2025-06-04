import streamlit as st
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
import io
import zipfile
import os
import re

# Load model once
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

# Stopwords to remove from caption
STOPWORDS = {"a", "an", "the", "with", "and", "on", "in", "of", "at", "to", "by", "for", "from"}

# App UI
st.set_page_config(page_title="AI Image Renamer")
st.title("🧠 AI-Powered T-shirt Graphic Renamer")
st.write("Upload your .png, .jpg, or .jpeg T-shirt graphics. AI will generate names and return a zip file with renamed files.")

uploaded_files = st.file_uploader("Upload image files", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

if uploaded_files and st.button("Generate & Download ZIP"):
    zip_buffer = io.BytesIO()
    total = len(uploaded_files)
    progress = st.progress(0, text="Starting...")

    with zipfile.ZipFile(zip_buffer, "w") as zip_file:
        for i, uploaded_file in enumerate(uploaded_files):
            # Open and resize image
            image = Image.open(uploaded_file).convert("RGB")
            image.thumbnail((512, 512))

            # Generate caption
            inputs = processor(images=image, return_tensors="pt").to(device)
            with torch.no_grad():
                output = model.generate(**inputs, max_length=20)
            raw_caption = processor.decode(output[0], skip_special_tokens=True)

            # Remove filler words and format filename
            # Remove filler words and format filename
            filtered_words = [
                word for word in re.findall(r"\w+", raw_caption.lower())
                if word not in STOPWORDS
            ]
            clean_caption = "_".join(filtered_words) or "image"
            clean_caption = clean_caption[:20].rstrip("_")  # Trim to 20 characters

            ext = os.path.splitext(uploaded_file.name)[1]
            new_filename = f"{clean_caption}{ext}"


            # Save image in zip
            image_bytes = io.BytesIO()
            image.save(image_bytes, format=image.format or "PNG")
            zip_file.writestr(new_filename, image_bytes.getvalue())

            # Update progress bar
            percent = (i + 1) / total
            progress.progress(percent, text=f"Processing {i + 1} of {total} images...")

    progress.empty()
    zip_buffer.seek(0)
    st.success("✅ All files processed successfully!")
    st.download_button(
        label="📥 Download ZIP with Renamed Images",
        data=zip_buffer,
        file_name="renamed_images.zip",
        mime="application/zip"
    )