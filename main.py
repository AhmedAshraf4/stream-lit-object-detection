import streamlit as st
from PIL import Image
import cv2
import numpy as np
from collections import Counter
from ultralytics import YOLO
import hashlib

model_dict = {
    1: "YOLOv8n.pt",
    2: "YOLOv8s.pt",
    3: "YOLOv8m.pt",
    4: "YOLOv8l.pt",
    5: "YOLOv8x.pt",
}


# Function to generate a consistent color based on the hash of the class name
def hash_color(class_name):
    hash_object = hashlib.sha256(class_name.encode())
    hex_dig = hash_object.hexdigest()
    r = int(hex_dig[0:2], 16)
    g = int(hex_dig[2:4], 16)
    b = int(hex_dig[4:6], 16)
    return r, g, b


# Function to predict classes and draw bounding boxes on the image
def predict_and_detect(chosen_model, img):
    model_results = chosen_model.predict(img)
    classes_detected = []
    for result in model_results:
        for box in result.boxes:
            class_name = result.names[int(box.cls[0])]
            classes_detected.append(class_name)
            color = hash_color(class_name)
            cv2.rectangle(img, (int(box.xyxy[0][0]), int(box.xyxy[0][1])),
                          (int(box.xyxy[0][2]), int(box.xyxy[0][3])), color, 2)
            cv2.putText(img, f"{class_name}",
                        (int(box.xyxy[0][0]), int(box.xyxy[0][1]) - 10),
                        cv2.FONT_HERSHEY_COMPLEX, 0.5, color, 1)
    return img, classes_detected


st.set_page_config(layout="wide")

container = st.container()

container.title("Stream-lit object detection")
container.write("Detect and analyze objects in images")

container.write("---")

col1, col2, col3 = container.columns([1, 11, 1])

with col1:
    st.write("Speed")

with col3:
    st.write("Accuracy")

with col2:
    level = st.slider("", min_value=1, max_value=5, value=3, step=1)

uploaded_file = container.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

col1, col2 = container.columns([3, 1])

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    with col1:
        placeholder = st.image(image, use_column_width=True)

    with col2:
        st.markdown(
            """
            <style>
            div.stButton > button
            {
            width: 100%;
            }
            </style>
            """,
            unsafe_allow_html=True
        )
        if st.button('Analyse image'):
            model = YOLO(model_dict[level])
            result_image, results = predict_and_detect(model, np.array(image))
            placeholder.empty()

            with col1:
                placeholder = st.image(Image.fromarray(result_image), use_column_width=True)
                st.write("Predicted with " + model_dict[level])

            class_counts = Counter(results)
            st.subheader("Objects and their counts:")
            for key, value in class_counts.items():
                st.write(f"{key}: {value}")

else:
    container.markdown(
        """
        <div style="border: 2px dashed #ddd; padding: 20px; text-align: center;">
            <h3>Upload Image</h3>
        </div>
        """,
        unsafe_allow_html=True
    )
