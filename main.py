import cv2
from ultralytics import YOLO
import numpy as np

model = YOLO("runs/detect/train6/weights/model.pt")

# blackjack value
BJ = {'A': 1, '2': 1, '3': 1, '4': 1, '5': 1, '6': 1, '7': 0,
      '8': 0, '9': 0, '10': -1, 'J': -1, 'Q': -1, 'K': -1}


def val(name):
    return BJ[name[:-1]]

class CardTracker:
    def __init__(self, max_disappeared=45, min_detection_count=10):
        ## card tracker for entire game
        ## max_disappeared is how many frames an object can be missing before being removed
        ## min_detection is minimum frames of detection before counting an object

        self.next_object_id = 0
        self.objects = {}  # id: (centroid, class_name, detection_count)
        self.disappeared = {} # dictionary that tracks how many frames each object has been missing
        self.max_disappeared = max_disappeared
        self.min_detection_count = min_detection_count  # require multiple detections before counting
        self.counted_ids = set() # set that keeps track of objects that have alreayd been counted

    def register(self, centroid, cls_name):
        self.objects[self.next_object_id] = (centroid, cls_name, 1)  # add new object with initial detection count of 1
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1

    def deregister(self, object_id):
        del self.objects[object_id]
        del self.disappeared[object_id]
        self.counted_ids.discard(object_id)  # remove from counted if present

    def update(self, rects):
        # if no detections, increment disappearance count for all objects
        if len(rects) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.objects

        # get centroids and names from detections
        input_centroids = np.zeros((len(rects), 2), dtype="int")
        input_names = []

        # calculate centroids of bounding box
        for i, (x1, y1, x2, y2, cls, name) in enumerate(rects):
            cx = int((x1 + x2) / 2.0)
            cy = int((y1 + y2) / 2.0)
            input_centroids[i] = (cx, cy)
            input_names.append(name)

        if len(self.objects) == 0:
            for i in range(len(input_centroids)):
                self.register(input_centroids[i], input_names[i])
        else:
            # get existing centroids and ids
            object_centroids = [obj[0] for obj in self.objects.values()]
            object_ids = list(self.objects.keys())

            # calculate distances between existing and new centroids
            D = np.linalg.norm(
                np.array(object_centroids)[:, np.newaxis] -
                input_centroids[np.newaxis, :], axis=2)

            # find the closest matches
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            used_row_indices = set()
            used_col_indices = set()

            # update existing objects
            for (row, col) in zip(rows, cols):
                if row in used_row_indices or col in used_col_indices:
                    continue

                object_id = object_ids[row]
                old_centroid, old_name, detection_count = self.objects[object_id]

                # update the object with new centroid and increment detection count
                new_centroid = input_centroids[col]
                new_name = input_names[col]
                self.objects[object_id] = (new_centroid, new_name, detection_count + 1)
                self.disappeared[object_id] = 0

                used_row_indices.add(row)
                used_col_indices.add(col)

            # handle unmatched existing objects
            unused_row_indices = set(range(0, D.shape[0])).difference(used_row_indices)
            unused_col_indices = set(range(0, D.shape[1])).difference(used_col_indices)

            # mark unmatched existing objects as disappeared
            for row in unused_row_indices:
                object_id = object_ids[row]
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)

            # register new objects that weren't matched
            for col in unused_col_indices:
                self.register(input_centroids[col], input_names[col])

        return self.objects

    def get_countable_objects(self):
        # return objects that have been detected enough times to be counted
        countable = {}
        for obj_id, (centroid, cls_name, detection_count) in self.objects.items():
            if detection_count >= self.min_detection_count and obj_id not in self.counted_ids:
                countable[obj_id] = (centroid, cls_name)
        return countable

    def mark_as_counted(self, obj_id):
        self.counted_ids.add(obj_id)


# initialize
cap = cv2.VideoCapture(0)
running_count = 0
tracker = CardTracker(max_disappeared=30, min_detection_count=3)  # Require 3 detections before counting


def filter_duplicate_detections(detections, iou_threshold=0.5):
    # Filter out duplicate detections of the same card based on IoU
    if len(detections) <= 1:
        return detections

    filtered_dets = []
    used_indices = set()

    for i, det1 in enumerate(detections):
        if i in used_indices:
            continue

        x1_1, y1_1, x2_1, y2_1, cls1, name1 = det1
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        keep = True

        for j, det2 in enumerate(detections[i + 1:], i + 1):
            if j in used_indices:
                continue

            x1_2, y1_2, x2_2, y2_2, cls2, name2 = det2

            # check if same class
            if cls1 == cls2:
                # calculate IoU
                x_left = max(x1_1, x1_2)
                y_top = max(y1_1, y1_2)
                x_right = min(x2_1, x2_2)
                y_bottom = min(y2_1, y2_2)

                if x_right > x_left and y_bottom > y_top:
                    intersection = (x_right - x_left) * (y_bottom - y_top)
                    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
                    iou = intersection / float(area1 + area2 - intersection)

                    if iou > iou_threshold:
                        used_indices.add(j)  # mark overlapping detection as duplicate

        if keep:
            filtered_dets.append(det1)

    return filtered_dets


while True:
    ret, frame = cap.read()
    if not ret:
        break

    # run YOLO detection
    results = model.predict(frame, conf=0.75, iou=0.5, verbose=False)[0]
    dets = [(int(b.xyxy[0][0]), int(b.xyxy[0][1]),
             int(b.xyxy[0][2]), int(b.xyxy[0][3]),
             int(b.cls), model.names[int(b.cls)]) for b in results.boxes]

    filtered_dets = dets

    # apply duplicate filtering
    filtered_dets = filter_duplicate_detections(filtered_dets, iou_threshold=0.4)

    # update tracker
    tracked_objects = tracker.update(filtered_dets)

    # get objects ready to be counted
    countable_objects = tracker.get_countable_objects()

    # count new objects
    for obj_id, (centroid, cls_name) in countable_objects.items():
        running_count += val(cls_name)
        tracker.mark_as_counted(obj_id)
        print(f"Counted: {cls_name} (ID: {obj_id})")

    # draw tracked objects
    for obj_id, (centroid, cls_name, detection_count) in tracked_objects.items():
        cx, cy = centroid
        # color based on detection count and whether counted
        if obj_id in tracker.counted_ids:
            color = (0, 255, 0)  # green for counted
        elif detection_count >= tracker.min_detection_count:
            color = (255, 255, 0)  # yellow for ready to count
        else:
            color = (0, 165, 255)  # orange for new detections

        cv2.circle(frame, (cx, cy), 4, color, -1)
        cv2.putText(frame, f"{cls_name} {obj_id} ({detection_count})", (cx - 10, cy - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # display running count
    cv2.putText(frame, f"Running: {round(running_count/2)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Card Tracker", frame)
    if cv2.waitKey(1) == 27:  # ESC key
        break

cap.release()
cv2.destroyAllWindows()