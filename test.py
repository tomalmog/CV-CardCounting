import cv2
from ultralytics import YOLO

# load trained model
model = YOLO("runs/detect/train5/weights/best.pt")  # Change path if needed

# start webcam
cap = cv2.VideoCapture(0)

# stop if no webcam was detected
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

	# plot the results, meaning draw the boxes and labels
	annotated_frame = results[0].plot()

	# display the annotated frame
	cv2.imshow("YOLOv8 Webcam Detection", annotated_frame)

	# press 'q' to quit
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

# release resources
cap.release()
cv2.destroyAllWindows()
