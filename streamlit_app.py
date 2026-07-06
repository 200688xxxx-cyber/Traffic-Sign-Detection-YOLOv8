from ultralytics import YOLO
import streamlit as st
from PIL import Image

st.title("🚦交通标志检测系统")

# 加载模型
model = YOLO("models/best.pt")

uploaded_file = st.file_uploader(
    "上传交通场景图片",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    st.write("图片已上传")

    image = Image.open(uploaded_file)

    st.image(image, caption="原始图片")

    st.write("开始检测...")

    results = model.predict(
        image,
        conf=0.25,
        verbose=False
    )

    st.write("检测完成")

    result_img = results[0].plot()

    st.image(
        result_img,
        caption="检测结果"
    )