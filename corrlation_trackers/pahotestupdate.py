import paho.mqtt.client as mqtt
import cv2
import time

# MQTT setup
broker = "192.168.1.61"  # Replace with the IP address of your broker
port = 1883
topic = "mouse/coordinates"

def on_connect(client, userdata, flags, rc):
    print("Connected with result code " + str(rc))

client = mqtt.Client()
client.on_connect = on_connect

client.connect(broker, port, 60)

# Variables to store coordinates and tracking state
first_click = None
second_click = None
roi = None
tracker_initialized = False

# Define the tracker
trackers = {
    'csrt' : cv2.legacy.TrackerCSRT_create,  # high accuracy, slow
    'mosse' : cv2.legacy.TrackerMOSSE_create,  # fast, low accuracy
    'kcf' : cv2.legacy.TrackerKCF_create,   # moderate accuracy and speed
    'medianflow' : cv2.legacy.TrackerMedianFlow_create,
    'mil' : cv2.legacy.TrackerMIL_create,
    'tld' : cv2.legacy.TrackerTLD_create,
    'boosting' : cv2.legacy.TrackerBoosting_create
}

tracker_key = 'kcf'
tracker = trackers[tracker_key]()

def on_click(event, x, y, flags, param):
    global first_click, second_click, roi, tracker_initialized, tracker, frame
    if event == cv2.EVENT_LBUTTONDOWN:
        if not first_click:
            first_click = (x, y)
            print(f"First click at: {first_click}")
        else:
            second_click = (x, y)
            print(f"Second click at: {second_click}")
            coordinates = {"x1": first_click[0], "y1": first_click[1], "x2": second_click[0], "y2": second_click[1]}
            client.publish(topic, str(coordinates))
            print(f"Rectangle coordinates sent: {coordinates}")

            # Initialize tracker with the selected ROI
            roi = (first_click[0], first_click[1], second_click[0] - first_click[0], second_click[1] - first_click[1])
            tracker.init(frame, roi)
            tracker_initialized = True
            first_click = None
            second_click = None

# Open the webcam
cap = cv2.VideoCapture(0)  # You can replace 0 with the appropriate camera index or video file

if not cap.isOpened():
    print("Error: Could not open video source.")
    exit()

cv2.namedWindow("Video Stream")
cv2.setMouseCallback("Video Stream", on_click)

fps = 0
start_time = time.time()

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        if tracker_initialized:
            success, box = tracker.update(frame)

            if success:
                x, y, w, h = [int(c) for c in box]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            else:
                print('Tracking failed')
                tracker_initialized = False
                tracker = trackers[tracker_key]()

        # Calculate and display FPS
        end_time = time.time()
        fps = 1 / (end_time - start_time)
        start_time = end_time
        cv2.putText(frame, f'FPS: {int(fps)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Video Stream", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
except KeyboardInterrupt:
    pass

# Cleanup
cap.release()
cv2.destroyAllWindows()
client.loop_stop()