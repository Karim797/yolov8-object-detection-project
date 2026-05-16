import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile
import os

st.set_page_config(
    page_title="YOLOv8 Object Detection",
    page_icon="📷",
    layout="wide"
)

st.title("YOLOv8 Object Detection Project")
st.write("Upload an image and the trained YOLOv8 model will detect objects.")

@st.cache_resource
def load_model():
    model = YOLO("best_model.pt")
    return model

model = load_model()

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

confidence = st.slider(
    "Confidence Threshold",
    min_value=0.1,
    max_value=1.0,
    value=0.25,
    step=0.05
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        st.image(image, use_container_width=True)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        image.save(temp_file.name)
        temp_path = temp_file.name

    results = model.predict(
        source=temp_path,
        conf=confidence,
        save=False
    )

    result_image = results[0].plot()

    with col2:
        st.subheader("Detection Result")
        st.image(result_image, use_container_width=True)

    os.remove(temp_path)

    st.success("Detection completed successfully.")
else:
    st.info("Please upload an image to start detection.")
