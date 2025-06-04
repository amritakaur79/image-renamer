import streamlit as st
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
import io
import zipfile
import os

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

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
            image = Image.open(uploaded_file)
            if image.mode != "RGB":
                image = image.convert("RGB")

            inputs = processor(images=image, return_tensors="pt").to(device)
            output = model.generate(**inputs)
            caption = processor.decode(output[0], skip_special_tokens=True).replace(" ", "_")

            ext = os.path.splitext(uploaded_file.name)[1]
            new_filename = f"{caption}{ext}"

            image_bytes = io.BytesIO()
            image.save(image_bytes, format=image.format or "PNG")
            zip_file.writestr(new_filename, image_bytes.getvalue())

            # Update progress
            progress = (i + 1) / total_files
            progress_bar.progress(progress, text=f"Processing {i + 1} of {total_files} images...")

    progress_bar.empty()  # Remove progress bar after completion
    zip_buffer.seek(0)

    st.success("All files processed successfully!")
    st.download_button(
        label="📥 Download Renamed Images ZIP",
        data=zip_buffer,
        file_name="renamed_images.zip",
        mime="application/zip"
    )
