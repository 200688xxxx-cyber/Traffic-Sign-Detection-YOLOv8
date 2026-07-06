if __name__ == "__main__":
    from ultralytics import YOLO

    model = YOLO("ultralytics/cfg/models/v8/yolov8s"
                 "_smallhead.yaml")
    model.load("yolov8s.pt")

    results = model.train(
        data=r"ultralytics/cfg/datasets/tt100k_small.yaml",
        epochs=300,
        imgsz=640,
        batch=8,
        workers=2,
        optimizer="AdamW",
        scale=0.2,
        device=0
    )