import gradio as gr
from PIL import Image
import os
import io
import zipfile
import torch
import re
from transformers import BlipProcessor, BlipForConditionalGeneration

# Load BLIP model once
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

STOPWORDS = {"a", "an", "the", "with", "and", "on", "in", "of", "at", "to", "by", "for", "from"}

def generate_caption(image):
    image = image.convert("RGB")
    inputs = processor(image, return_tensors="pt")
    out = model.generate(**inputs)
    return processor.decode(out[0], skip_special_tokens=True)

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

def process_images(images):
    existing_names = set()
    zip_buffer = io.BytesIO()
    renamed_files = []

    with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED) as zipf:
        for image in images:
            caption = generate_caption(image)
            filename = clean_caption(caption, existing_names) + ".png"
            img_bytes = io.BytesIO()
            image.save(img_bytes, format="PNG")
            zipf.writestr(filename, img_bytes.getvalue())
            renamed_files.append((filename, image))

    zip_buffer.seek(0)
    return zip_buffer

iface = gr.Interface(
    fn=process_images,
    inputs=gr.File(file_types=[".png", ".jpg", ".jpeg"], label="Upload T-Shirt Graphics", file_count="multiple", type="pil"),
    outputs=gr.File(label="Download Renamed Zip"),
    title="🧠 AI T-Shirt Graphic Renamer",
    description="Upload multiple T-shirt graphic images. This app uses BLIP to analyze and rename each graphic with meaningful names, and returns a ZIP of the renamed files.",
    allow_flagging="never"
)

if __name__ == "__main__":
    iface.launch()
