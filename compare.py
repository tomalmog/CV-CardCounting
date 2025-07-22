from ultralytics import YOLO

def main():
    # load the models
    model2 = YOLO("runs/detect/train4/weights/best.pt")
    model3 = YOLO("runs/detect/train5/weights/best.pt")

    # evaluate each on the same dataset
    metrics2 = model2.val(data='cards_yolo_split/data.yaml')
    metrics3 = model3.val(data='cards_yolo_split/data.yaml')

if __name__ == "__main__":
    main()