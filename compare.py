from ultralytics import YOLO

def main():
    # load the models
    model = YOLO("colab_model.pt")
    model2 = YOLO("runs/detect/train6/weights/best.pt")

    # evaluate each on the same dataset
    metrics = model.val(data='kaggle_set/data.yaml')
    metrics2 = model2.val(data='kaggle_set/data.yaml')

if __name__ == "__main__":
    main()