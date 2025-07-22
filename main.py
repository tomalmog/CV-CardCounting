from ultralytics import YOLO

def main():
    model = YOLO("yolov8n.pt")  # or yolov8s.pt, etc.
    model.train(
        data="cards_yolo_split/data.yaml",
        epochs=100,
        imgsz=640,
        batch=8,
        device=0
    )

if __name__ == "__main__":
    main()


# train with
# yolo detect train model=runs/detect/train/weights/last.pt data=cards_yolo_split/data.yaml epochs=X

