import time
import tempfile
import os

import streamlit as st
from ultralytics import YOLO
from PIL import Image


st.set_page_config(
    page_title="YOLOv8 Object Detection",
    page_icon="🤖",
    layout="wide"
)

st.title("AI-Powered Object Detection using YOLOv8")
st.write(
    "Upload an image and the AI model will detect and classify objects in real time."
)


with st.sidebar:
    st.header("Model Information")
    st.write("Model: YOLOv8 Fine-Tuned")
    st.write("Dataset: COCO + Manual Labels")
    st.write("Framework: Ultralytics YOLO")
    st.write("Deployment: Streamlit")

    st.markdown("---")
    confidence = st.slider(
        "Confidence Threshold",
        min_value=0.10,
        max_value=1.00,
        value=0.30,
        step=0.05
    )


@st.cache_resource
def load_model():
    return YOLO("best_model.pt")


model = load_model()


uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
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

    start_time = time.time()

    results = model.predict(
        source=temp_path,
        conf=confidence,
        save=False
    )

    end_time = time.time()
    processing_time = end_time - start_time

    result_image = results[0].plot()

    detected_objects = len(results[0].boxes)

    with col2:
        st.subheader("Detection Result")
        st.image(result_image, use_container_width=True)

    st.markdown("## Detection Summary")

    metric1, metric2, metric3 = st.columns(3)

    metric1.metric("Detected Objects", detected_objects)
    metric2.metric("Processing Time", f"{processing_time:.2f} sec")
    metric3.metric("Confidence Threshold", f"{confidence:.2f}")

    if detected_objects > 0:
        st.subheader("Detected Classes")

        class_ids = results[0].boxes.cls.tolist()
        class_names = results[0].names

        detected_class_names = [class_names[int(cls_id)] for cls_id in class_ids]

        for i, name in enumerate(detected_class_names, start=1):
            st.write(f"{i}. {name}")
    else:
        st.warning("No objects detected. Try lowering the confidence threshold.")

    os.remove(temp_path)

    st.success("Detection completed successfully.")

else:
    st.info("Please upload an image to start detection.")
