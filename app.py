import streamlit as st
from transformers import BlipProcessor, BlipForConditionalGeneration
from transformers.pipelines import pipeline
from PIL import Image
import torch
import time
import random

import streamlit as st

def main():
    st.title("Multi-Modal AI Story Generator")
    # other UI code...

# Load BLIP Model for image + text understanding
@st.cache_resource
def load_blip():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

# Load text generation pipeline
@st.cache_resource
def load_generator():
    return pipeline("text-generation", model="gpt2")

processor, model = load_blip()
text_generator = load_generator()

# ---- UI CONFIG ----
st.set_page_config(page_title="Procedural Narrative Generator", layout="wide")
st.markdown(
    """
    <style>
    body { background-color: #0f0f0f; color: #fdd835; }
    .stApp { background-color: #0f0f0f; color: #fdd835; }
    h1, h2, h3 { color: #fdd835; }
    .title-text { font-size: 3em; animation: fadeIn 2s; }
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("<h1 class='title-text'>🎮 Procedural Narrative Generator</h1>", unsafe_allow_html=True)

# ---- USER INPUT ----
col1, col2 = st.columns(2)
with col1:
    story_seed = st.text_input("Enter your game world idea or a story seed:", "A knight in a ruined future kingdom")

with col2:
    image_file = st.file_uploader("Upload an image for the scene (optional)", type=["jpg", "png"])

# ---- STORY + IMAGE CAPTIONING ----
if st.button("🪄 Generate Story"):
    with st.spinner("Generating your narrative..."):
        time.sleep(1)

        # Generate story
        generated = text_generator(story_seed, max_length=120, num_return_sequences=1)[0]['generated_text']

        with st.expander("📖 Click to reveal your generated story", expanded=True):
            st.markdown(f"**Narrative:**\n\n{generated}")

        if image_file:
            image = Image.open(image_file).convert('RGB')
            st.image(image, caption="Uploaded Scene", use_column_width=True)

            inputs = processor(images=image, return_tensors="pt")
            pixel_values = inputs['pixel_values']

            output_ids = model.generate(pixel_values=pixel_values)
            caption = processor.decode(output_ids[0], skip_special_tokens=True)

            st.success(f"🖼️ Scene Description (BLIP): {caption}")
            st.info(f"🔁 Story Twist: What if {caption.lower()} changed the fate of the kingdom?")




# ---- FOOTER ----
st.markdown("---")
st.caption("Made with ❤️ using Streamlit + Transformers. Theme: Black & Yellow. Team: Shubham, Sharvari, Shrawani, Rohini, Vansh.")

if __name__ == "__main__":
    main()
