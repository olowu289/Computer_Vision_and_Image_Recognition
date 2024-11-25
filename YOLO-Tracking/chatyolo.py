import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# Desired video properties
DESIRED_WIDTH = 1280
DESIRED_HEIGHT = 720
DESIRED_FPS = 30


# Load YOLOv8 model pre-trained on COCO dataset
model = YOLO("yolov8n.pt")

# Modify the model to track only the 'person' class
person_class_id = 0  # COCO dataset class ID for 'person'
model.classes = [person_class_id]

# Initialize Deep SORT
tracker = DeepSort(max_age=30, n_init=3)

# Open video capture (0 for webcam, or provide video file path)
cap = cv2.VideoCapture(0)

# Setup GStreamer pipeline for video streaming
GSTREAMER_PIPELINE = (
    'appsrc ! videoconvert ! x264enc speed-preset=ultrafast tune=zerolatency ! '
    'rtph264pay config-interval=1 pt=96 ! udpsink host=192.168.1.12 port=5600'
)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, DESIRED_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DESIRED_HEIGHT)
cap.set(cv2.CAP_PROP_FPS, DESIRED_FPS)

out = cv2.VideoWriter(GSTREAMER_PIPELINE, cv2.CAP_GSTREAMER, 0, DESIRED_FPS, (DESIRED_WIDTH, DESIRED_HEIGHT), True)

if not out.isOpened():
    print("Failed to open GStreamer pipeline.")
    cap.release()
    exit()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Perform YOLOv8 object detection
    results = model(frame)

    # Extract bounding boxes and confidence scores
    detections = []
    for result in results[0].boxes:
        # Check if the detected class is 'person'
        if int(result.cls[0]) == person_class_id:
            bbox = result.xyxy[0].cpu().numpy()  # Extract bounding box (x1, y1, x2, y2)
            conf = result.conf[0].cpu().numpy()  # Extract confidence
            detections.append([bbox, conf])

    # Update tracker with person detections
    tracks = tracker.update_tracks(detections, frame=frame)

    # Draw bounding boxes and track IDs on the frame
    for track in tracks:
        if not track.is_confirmed():
            continue
        bbox = track.to_tlbr()  # Get bounding box coordinates in (x1, y1, x2, y2)
        track_id = track.track_id  # Get track ID
        cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
        cv2.putText(frame, f"ID: {track_id}", (int(bbox[0]), int(bbox[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Send the frame via GStreamer
    out.write(frame)

    # Display the frame locally
    cv2.imshow("YOLOv8 + Deep SORT", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
