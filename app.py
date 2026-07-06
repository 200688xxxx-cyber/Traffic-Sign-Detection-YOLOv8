from ultralytics import YOLO
import gradio as gr

# 加载模型
model = YOLO("models/best.pt")

# 检测函数
def detect(image):
    results = model.predict(
        source=image,
        conf=0.25
    )

    annotated_img = results[0].plot()

    return annotated_img

# Demo界面
demo = gr.Interface(
    fn=detect,
    inputs=gr.Image(type="numpy"),
    outputs=gr.Image(type="numpy"),
    title="🚦 Traffic Sign Detector",
    description="上传交通场景图片，自动识别交通标志",
)

if __name__ == "__main__":
    demo.launch(
        share=True
    )