import cv2
import threading
import paho.mqtt.client as mqtt
import time
import numpy as np
import concurrent.futures
import dlib

import asyncio
from mavsdk import System

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

# Default bounding box size
DEFAULT_WIDTH = 150
DEFAULT_HEIGHT = 200

# GStreamer pipeline configuration
GSTREAMER_PIPELINE = (
    'appsrc ! videoconvert ! x264enc speed-preset=ultrafast tune=zerolatency ! '
    'rtph264pay config-interval=1 pt=96 ! udpsink host=192.168.1.61 port=5600'
)

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)  # ORB matcher

async def setup_drone():
    drone = System()
    # Update the connection address to the serial port of the Pixhawk
    await drone.connect(system_address="serial:///dev/ttyUSB0:57600")  # Adjust the baud rate if needed
    print("Waiting for drone to connect...")
    async for state in drone.core.connection_state():
        if state.is_connected:
            print("Drone connected!")
            break
    return drone



async def align_drone_to_target(drone, center_x, center_y, frame_width, frame_height):
    x_offset = (center_x - frame_width // 2) / (frame_width // 2)
    y_offset = (center_y - frame_height // 2) / (frame_height // 2)

    # Yaw (horizontal alignment) and pitch (vertical alignment)
    yaw_adjustment = -x_offset * 10  # Adjust sensitivity as needed
    pitch_adjustment = -y_offset * 10

    # Send gimbal adjustments
    await drone.gimbal.set_pitch_and_yaw(pitch_adjustment, yaw_adjustment)



# Function to calculate adaptive window size based on object size
def calculate_adaptive_window_size(bbox, min_window=30, max_window=200):
    object_width, object_height = bbox[2], bbox[3]
    object_size = (object_width + object_height) // 2
    window_size = int(object_size * 1.5)
    window_size = max(min_window, min(max_window, window_size))
    return window_size

# Global sliding window generation with adaptive window size and step size
def generate_adaptive_sliding_windows(frame_width, frame_height, bbox, step_size_factor=0.25):
    window_size = calculate_adaptive_window_size(bbox)
    step_size = max(1, int(window_size * step_size_factor))
    bboxes = []
    for y in range(0, frame_height - window_size, step_size):
        for x in range(0, frame_width - window_size, step_size):
            bboxes.append((x, y, window_size, window_size))
    return bboxes

# Function to check if a point is inside a rectangle
def is_point_inside_rectangle(x, y, rect):
    rx, ry, rw, rh = rect
    return rx <= x <= x + rw and ry <= y <= ry + rh

# Function to extract high pixel intensity region within the bounding box
def extract_high_pixel_region(frame, bbox, threshold=100):
    x, y, w, h = [int(v) for v in bbox]
    if x < 0 or y < 0 or x + w > frame.shape[1] or y + h > frame.shape[0]:
        return None, None, None
    roi = frame[y:y+h, x:x+w]
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, high_pixel_mask = cv2.threshold(gray_roi, threshold, 255, cv2.THRESH_BINARY_INV)
    high_pixel_values = gray_roi[high_pixel_mask == 255]
    return high_pixel_mask, roi, high_pixel_values

# Modified compute_color_histogram to only focus on high pixel intensity region
def compute_color_histogram(image, bbox):
    high_pixel_mask, roi, _ = extract_high_pixel_region(image, bbox)
    if high_pixel_mask is None or roi is None:
        return None
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], high_pixel_mask, [180, 256], [0, 180, 0, 256])
    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
    return hist

# Modified compute_ORB_features to only focus on keypoints in the high pixel intensity region
def compute_ORB_features(image, bbox):
    high_pixel_mask, roi, _ = extract_high_pixel_region(image, bbox)
    if high_pixel_mask is None or roi is None:
        return None, None
    orb = cv2.ORB_create()
    keypoints, descriptors = orb.detectAndCompute(roi, high_pixel_mask)
    return keypoints, descriptors

# Function to compare histograms
def compare_histograms(hist1, hist2):
    return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)

# Function to search the entire frame for re-identification
def search_entire_frame(frame, initial_hist, initial_keypoints, initial_descriptors, bf):
    frame_hist = compute_color_histogram(frame, (0, 0, frame.shape[1], frame.shape[0]))
    frame_keypoints, frame_descriptors = compute_ORB_features(frame, (0, 0, frame.shape[1], frame.shape[0]))
    if frame_hist is None or frame_descriptors is None:
        return None
    similarity = compare_histograms(initial_hist, frame_hist)
    matches = bf.knnMatch(initial_descriptors, frame_descriptors, k=2)
    good_matches = [m for m, n in matches if m.distance < 0.85 * n.distance]
    if similarity > 0.75 and len(good_matches) > 5:
        return (0, 0, frame.shape[1], frame.shape[0])
    return None

# Sliding window search using parallel processing
def parallel_search(frame, bboxes, initial_hist, initial_keypoints, initial_descriptors, bf):
    found_bbox = None
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(search_entire_frame, frame, initial_hist, initial_keypoints, initial_descriptors, bf) for bbox in bboxes]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result is not None:
                found_bbox = result
                break
    return found_bbox

# Re-identification thread that now searches across the entire frame
def re_identification_thread():
    global rectangle, reidentifying, tracker, tracking_initialized, initial_hist, initial_keypoints, initial_descriptors
    print("Re-identification thread started.")
    
    frame_width, frame_height = DESIRED_WIDTH, DESIRED_HEIGHT
    while True:
        with frame_lock:
            current_frame = frame.copy()
        if orb_active and initial_descriptors is not None:
            sliding_window_bboxes = generate_adaptive_sliding_windows(frame_width, frame_height, rectangle)
            found_bbox = parallel_search(current_frame, sliding_window_bboxes, initial_hist, initial_keypoints, initial_descriptors, bf)
            if found_bbox:
                print("Re-identification successful.")
                tracker = dlib.correlation_tracker()
                tracker.start_track(current_frame, tuple(found_bbox))
                initial_hist = compute_color_histogram(current_frame, found_bbox)
                initial_keypoints, initial_descriptors = compute_ORB_features(current_frame, found_bbox)
                tracking_initialized = True
                with lock:
                    reidentifying = False
                break
        time.sleep(0.1)
    print("Re-identification thread stopped.")

# Function to change color after a delay
def change_color_after_delay():
    global color, last_command_time, lock, orb_active, object_keypoints, object_descriptors, initial_hist, initial_keypoints, initial_descriptors
    while True:
        time.sleep(1)
        with lock:
            if last_command_time and (time.time() - last_command_time > 5):
                color = (0, 255, 0)
                if not orb_active and rectangle is not None:
                    x, y, w, h = rectangle
                    initial_keypoints, initial_descriptors = compute_ORB_features(frame, rectangle)
                    initial_hist = compute_color_histogram(frame, rectangle)
                    orb_active = True
                    print("ORB feature extraction activated.")

# Mouse callback function to get the x and y coordinates
def mouse_click_event(event, x, y, flags, param):
    global rectangle, tracker, tracking_initialized, last_command_time, color, lock

    if event == cv2.EVENT_LBUTTONDOWN:  # Left button click
        with lock:
            rectangle = (x - DEFAULT_WIDTH // 2, y - DEFAULT_HEIGHT // 2, DEFAULT_WIDTH, DEFAULT_HEIGHT)
            print(f"Mouse clicked at: ({x}, {y})")
            print(f"Rectangle set: {rectangle}")
            tracker = dlib.correlation_tracker()
            dlib_rect = dlib.rectangle(rectangle[0], rectangle[1], rectangle[0] + rectangle[2], rectangle[1] + rectangle[3])
            tracker.start_track(frame, dlib_rect)
            tracking_initialized = True
            last_command_time = time.time()
            color = (255, 0, 0)

# Video feed thread to process the video and tracking
def video_feed_thread(drone):
    global rectangle, tracker, tracking_initialized, color, frame, initial_keypoints, initial_descriptors, orb_active, lost, initial_hist, reidentifying, reid_thread

    cap = cv2.VideoCapture("/dev/video0")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, DESIRED_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DESIRED_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, DESIRED_FPS)

    out = cv2.VideoWriter(GSTREAMER_PIPELINE, cv2.CAP_GSTREAMER, 0, DESIRED_FPS, (DESIRED_WIDTH, DESIRED_HEIGHT), True)

    if not out.isOpened():
        print("Failed to open GStreamer pipeline.")
        return

    prev_time = time.time()
    fps = 0

    cv2.namedWindow("Video Feed")
    cv2.setMouseCallback("Video Feed", mouse_click_event)

    # Asynchronous alignment loop
    async def align_loop():
        while True:
            if tracking_initialized and rectangle:
                x, y, w, h = rectangle
                center_x = x + w // 2
                center_y = y + h // 2
                # Align the drone to the target
                await align_drone_to_target(drone, center_x, center_y, DESIRED_WIDTH, DESIRED_HEIGHT)
            await asyncio.sleep(0.1)  # Adjust delay for smoother alignment

    # Start the asynchronous alignment loop in the event loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.create_task(align_loop())

    while True:
        ret, frame = cap.read()
        if ret:
            current_time = time.time()
            fps = 1 / (current_time - prev_time)
            prev_time = current_time

            with frame_lock:
                if rectangle is not None:
                    if not tracking_initialized:
                        x, y, w, h = rectangle
                        dlib_rect = dlib.rectangle(x, y, x + w, y + h)
                        tracker.start_track(frame, dlib_rect)
                        tracking_initialized = True
                    confidence = tracker.update(frame)
                    if confidence > 5:
                        pos = tracker.get_position()
                        x = int(pos.left())
                        y = int(pos.top())
                        width = int(pos.width())
                        height = int(pos.height())
                        cv2.rectangle(frame, (x, y), (x + width, y + height), color, 3)
                        center_x = x + width // 2
                        center_y = y + height // 2
                        cv2.circle(frame, (center_x, center_y), 1, (0, 255, 0), 1)
                    else:
                        print("Tracking failed. Starting re-identification process.")
                        if not reidentifying:
                            with lock:
                                reidentifying = True
                            reid_thread = threading.Thread(target=re_identification_thread, daemon=True)
                            reid_thread.start()

            cv2.putText(frame, f"FPS: {fps:.2f}", (100, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, "dlib correlation Tracker", (100, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            out.write(frame)
            cv2.imshow("Video Feed", frame)

        if cv2.waitKey(33) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # Stop the event loop when the video thread exits
    loop.stop()



# Initialize the rest of your system
threading.Thread(target=change_color_after_delay, daemon=True).start()

client = mqtt.Client()
client.connect("192.168.1.61", 1883, 60)

async def main():
    # Step 1: Setup the drone connection using MavSDK
    drone = await setup_drone()

    # Step 2: Start the video feed thread with the drone passed as an argument
    video_thread = threading.Thread(target=video_feed_thread, args=(drone,), daemon=True)
    video_thread.start()

    # Step 3: Keep the asyncio loop alive for any other async tasks
    while True:
        await asyncio.sleep(1)

# Entry point of the program
if __name__ == "__main__":
    asyncio.run(main())
