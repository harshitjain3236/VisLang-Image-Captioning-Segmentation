# app.py
import streamlit as st
from PIL import Image
from caption_segmentation import generate_caption, segment_image
import io

st.set_page_config(page_title="Awesome Image Captioning & Segmentation", layout="wide")

st.markdown("""
    <h1 style='text-align: center; color: #4B0082;'>🎉 Image Captioning & Segmentation App 🎉</h1>
    <p style='text-align: center;'>Upload an image and see AI magic!</p>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    with st.spinner("Generating caption... ⏳"):
        caption = generate_caption(image)
    
    with st.spinner("Segmenting image... ⏳"):
        segmented = segment_image(image)
    
    st.success("Done! ✅")
    st.markdown(f"**Generated Caption:** {caption}")
    
    st.image(segmented, caption="Segmented Image", use_column_width=True)
    
    # Download button
    buf = io.BytesIO()
    segmented.save(buf, format="PNG")
    st.download_button(
        label="Download Segmented Image",
        data=buf.getvalue(),
        file_name="segmented.png",
        mime="image/png"
    )
