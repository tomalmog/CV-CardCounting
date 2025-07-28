import cv2
from ultralytics import YOLO

# load trained model
model = YOLO("runs/detect/train4/weights/best.pt")  

# start webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
	print("Cannot open webcam")
	exit()

while True:
	# read a frame from the webcam
	ret, frame = cap.read()
	if not ret:
		break

	# run model inference on the frame
	results = model(frame, conf=0.1)

	# draw the boxes and labels
	annotated_frame = results[0].plot()

	# display the annotated frame
	cv2.imshow("YOLOv8 Webcam Detection", annotated_frame)

	# quit with q
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()

### yolo detect train model=runs/detect/train4/weights/last.pt data=cards_yolo_split/data.yaml epochs=8 imgsz=640 batch=8 workers=4 optimizer=AdamW patience=15
### yolo detect train model=colab_model.pt data=kaggle_set/data.yaml epochs=8 imgsz=640 batch=8 workers=4 optimizer=AdamW patience=15