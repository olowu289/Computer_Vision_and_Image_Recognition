import cv2
import numpy as np
import gi
import os
from gi.repository import Gst, GLib

gi.require_version('Gst', '1.0')

# Setting up our tracker
trackers = {
    'csrt': cv2.legacy.TrackerCSRT_create,  # high accuracy, slow
    'mosse': cv2.legacy.TrackerMOSSE_create,  # fast, low accuracy
    'kcf': cv2.legacy.TrackerKCF_create,   # moderate accuracy and speed
    'medianflow': cv2.legacy.TrackerMedianFlow_create,
    'mil': cv2.legacy.TrackerMIL_create,
    'tld': cv2.legacy.TrackerTLD_create,
    'boosting': cv2.legacy.TrackerBoosting_create
}

tracker_key = 'csrt'
roi = None
tracker = trackers[tracker_key]()

def select_roi(frame):
    global roi, tracker
    roi = cv2.selectROI('Tracking', frame, fromCenter=False, showCrosshair=True)
    tracker.init(frame, roi)
    cv2.destroyWindow('Tracking')

# Start capturing video
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

# Get the width and height of the frame from the capture device
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

print(f"Width: {width}, Height: {height}, FPS: {fps}")

# Initialize GStreamer
Gst.init(None)

# Define the GStreamer pipeline for video streaming
pipeline_str = (
    "appsrc ! videoconvert ! "
    "x264enc tune=zerolatency ! "
    "rtph264pay config-interval=1 pt=96 ! "
    "udpsink host=192.168.1.61 port=5600"
)

pipeline = Gst.parse_launch(pipeline_str)

# Start the GStreamer pipeline
pipeline.set_state(Gst.State.PLAYING)

# Define the sharpening kernel
kernel = np.array([[0, -1, 0],
                   [-1, 5, -1],
                   [0, -1, 0]])

def push_frame_to_pipeline(frame):
    # Convert the frame to GstBuffer and push it to the pipeline
    data = frame.tobytes()
    buf = Gst.Buffer.new_allocate(None, len(data), None)
    buf.fill(0, data)

    # Get the appsrc element and push the buffer to it
    appsrc = pipeline.get_by_name("appsrc")
    appsrc.emit("push-buffer", buf)

# Add a callback to handle the end of stream
def on_eos(bus, msg):
    print('End-Of-Stream reached')
    loop.quit()

bus = pipeline.get_bus()
bus.add_signal_watch()
bus.connect("message::eos", on_eos)

# Create a GLib Main Loop and set it to run
loop = GLib.MainLoop()

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Apply sharpening effect
        sharpened_frame = cv2.filter2D(frame, -1, kernel)

        if roi is not None:
            success, box = tracker.update(sharpened_frame)
            if success:
                x, y, w, h = [int(c) for c in box]
                cv2.rectangle(sharpened_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            else:
                print("Tracking Failed")
                roi = None
                tracker = trackers[tracker_key]()

        # Push frame to GStreamer pipeline
        push_frame_to_pipeline(sharpened_frame)

        # Handle keyboard input without cv2.waitKey
        # For demonstration purposes, let's use a timer to check for 's' and 'q' inputs
        # This part can be customized as needed for your use case

except KeyboardInterrupt:
    pass

# Cleanup
cap.release()
pipeline.set_state(Gst.State.NULL)
cv2.destroyAllWindows()
