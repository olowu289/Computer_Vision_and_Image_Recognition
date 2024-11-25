import cv2
import threading
import paho.mqtt.client as mqtt
import time
import numpy as np
import concurrent.futures
import dlib

# Desired video properties
DESIRED_WIDTH = 640
DESIRED_HEIGHT = 480
DESIRED_FPS = 30

# Global variables for a single target
rectangle = None
tracker = None
tracking_initialized = False
last_command_time = None
color = (255, 0, 0)  # Initial color (red)
lock = threading.Lock()
object_keypoints = None
object_descriptors = None
initial_hist = None  # For color histogram
orb_active = False
lost = False
reidentifying = False  # Flag to track if re-identification is running
reid_thread = None  # Re-identification thread
frame_lock = threading.Lock()  # Lock for the frame

# GStreamer pipeline configuration
GSTREAMER_PIPELINE = (
    'appsrc ! videoconvert ! x264enc speed-preset=ultrafast tune=zerolatency ! '
    'rtph264pay config-interval=1 pt=96 ! udpsink host=192.168.1.61 port=5600'
)

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)  # ORB matcher

# Function to handle mouse events
def mouse_click(event, x, y, flags, param):
    global rectangle, tracker, tracking_initialized
    if event == cv2.EVENT_LBUTTONDOWN:
        # Set the midpoint as the click location
        width = 200
        height = 150
        x_start = x - width // 2
        y_start = y - height // 2
        rectangle = (x_start, y_start, width, height)
        print(f"Clicked at ({x}, {y}), setting rectangle: {rectangle}")

        # Reset the tracker and tracking status
        tracker = dlib.correlation_tracker()
        tracking_initialized = False

# MQTT connection and message handling functions
def on_connect(client, userdata, flags, rc):
    print(f"Connected with result code {rc}")
    client.subscribe("drone/com")

def on_message(client, userdata, msg):
    global rectangle, tracker, tracking_initialized, last_command_time, color, lock
    try:
        payload = msg.payload.decode()
        commands = payload.split('\n')
        for command in commands:
            command = command.strip()
            if not command:
                continue
            # Handle commands as needed
            print(f"Command received: {command}")
    except Exception as e:
        print(f"Error processing message: {e}")

# Video feed thread to process the video and tracking
def video_feed_thread():
    global rectangle, tracker, tracking_initialized, color, frame, initial_keypoints, initial_descriptors, orb_active, lost, initial_hist, reidentifying, reid_thread

    cap = cv2.VideoCapture("/dev/video0")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, DESIRED_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DESIRED_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, DESIRED_FPS)

    out = cv2.VideoWriter(GSTREAMER_PIPELINE, cv2.CAP_GSTREAMER, 0, DESIRED_FPS, (DESIRED_WIDTH, DESIRED_HEIGHT), True)

    if not out.isOpened():
        print("Failed to open GStreamer pipeline.")
        return

    # Initialize variables for FPS calculation
    prev_time = time.time()
    fps = 0

    # Set up mouse callback
    cv2.namedWindow("Video Feed")
    cv2.setMouseCallback("Video Feed", mouse_click)

    while True:
        ret, frame = cap.read()
        if ret:
            # Calculate FPS
            current_time = time.time()
            fps = 1 / (current_time - prev_time)
            prev_time = current_time

            with frame_lock:
                if rectangle is not None:
                    if not tracking_initialized:
                        # Convert (x, y, width, height) to dlib.rectangle
                        x, y, w, h = rectangle
                        dlib_rect = dlib.rectangle(x, y, x + w, y + h)
                        tracker.start_track(frame, dlib_rect)  # Use dlib.rectangle
                        tracking_initialized = True
                    # Update the tracker
                    confidence = tracker.update(frame)
                    if confidence > 5:  # Threshold for successful tracking (adjust as needed)
                        # Get the new position of the tracked object
                        pos = tracker.get_position()
                        x = int(pos.left())
                        y = int(pos.top())
                        width = int(pos.width())
                        height = int(pos.height())

                        # Draw the updated rectangle and center point
                        cv2.rectangle(frame, (x, y), (x + width, y + height), color, 3)
                        center_x = x + width // 2
                        center_y = y + height // 2
                        cv2.circle(frame, (center_x, center_y), 1, (0, 255, 0), 1)
                    else:
                        print("Tracking failed.")
                        
            # Display FPS on the frame
            cv2.putText(frame, f"FPS: {fps:.2f}", (100, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow("Video Feed", frame)
            out.write(frame)

        if cv2.waitKey(33) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

# Initialize the rest of your system
threading.Thread(target=video_feed_thread, daemon=True).start()

client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message
client.connect("192.168.1.61", 1883, 60)

client.loop_forever()
