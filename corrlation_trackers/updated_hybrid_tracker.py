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
dlib_tracker = None
kcf_tracker = None
tracking_initialized = False
last_command_time = None
lock = threading.Lock()
object_keypoints = None
object_descriptors = None
initial_hist = None  # For color histogram
orb_active = False
lost = False
reidentifying = False  # Flag to track if re-identification is running
reid_thread = None  # Re-identification thread
frame_lock = threading.Lock()  # Lock for the frame
using_kcf = False  # Flag for using KCF tracker
kcf_start_time = None  # Time when KCF started
last_bbox_size = None  # Store the last known bounding box size (width, height)
dlib_failure_count = 0  # Counter for dlib tracker failure

# GStreamer pipeline configuration
GSTREAMER_PIPELINE = (
    'appsrc ! videoconvert ! x264enc speed-preset=ultrafast tune=zerolatency ! '
    'rtph264pay config-interval=1 pt=96 ! udpsink host=192.168.1.61 port=5600'
)

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)  # ORB matcher

# Function to convert percentage to pixel coordinates based on screen resolution
def get_pixel_coordinates(x_percent, y_percent, width_percent, height_percent, feed_width=DESIRED_WIDTH, feed_height=DESIRED_HEIGHT):
    center_x = int((x_percent / 100) * feed_width)
    center_y = int((y_percent / 100) * feed_height)
    width_pixel = int((width_percent / 100) * feed_width)
    height_pixel = int((height_percent / 100) * feed_height)
    x_pixel = center_x - width_pixel // 2
    y_pixel = center_y - height_pixel // 2
    return x_pixel, y_pixel, width_pixel, height_pixel

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

    # Extract the ROI (region of interest) from the original frame
    roi = frame[y:y+h, x:x+w]

    # Convert the ROI to grayscale for intensity analysis
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # Apply a threshold to extract high pixel values (darker regions)
    _, high_pixel_mask = cv2.threshold(gray_roi, threshold, 255, cv2.THRESH_BINARY_INV)

    # Get pixel intensity values
    high_pixel_values = gray_roi[high_pixel_mask == 255]

    # Return both the grayscale mask and original ROI
    return high_pixel_mask, roi, high_pixel_values

# Modified compute_color_histogram to only focus on high pixel intensity region
def compute_color_histogram(image, bbox):
    high_pixel_mask, roi, _ = extract_high_pixel_region(image, bbox)
    
    if high_pixel_mask is None or roi is None:
        return None

    # Compute the histogram only in the high pixel intensity region
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], high_pixel_mask, [180, 256], [0, 180, 0, 256])
    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
    return hist

# Modified compute_ORB_features to only focus on keypoints in the high pixel intensity region
def compute_ORB_features(image, bbox):
    high_pixel_mask, roi, _ = extract_high_pixel_region(image, bbox)
    
    if high_pixel_mask is None or roi is None:
        return None, None

    # Detect ORB keypoints and descriptors in the high pixel intensity region
    orb = cv2.ORB_create()
    keypoints, descriptors = orb.detectAndCompute(roi, high_pixel_mask)
    
    return keypoints, descriptors

# Function to compare histograms
def compare_histograms(hist1, hist2):
    return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)

# Function to search the entire frame for re-identification
def search_entire_frame(frame, initial_hist, initial_keypoints, initial_descriptors, bf):
    # Compute color histogram for the entire frame
    frame_hist = compute_color_histogram(frame, (0, 0, frame.shape[1], frame.shape[0]))  # Full frame
    frame_keypoints, frame_descriptors = compute_ORB_features(frame, (0, 0, frame.shape[1], frame.shape[0]))  # Full frame

    # Check if histogram or descriptors are invalid
    if frame_hist is None or frame_descriptors is None:
        return None

    # Compare histograms
    similarity = compare_histograms(initial_hist, frame_hist)

    # Use a relaxed matching criteria with ORB features
    matches = bf.knnMatch(initial_descriptors, frame_descriptors, k=2)
    good_matches = [m for m, n in matches if m.distance < 0.85 * n.distance]

    # Reduced similarity threshold and match count for re-identification
    if similarity > 0.75 and len(good_matches) > 5:
        return (0, 0, frame.shape[1], frame.shape[0])  # Return entire frame as the bounding box

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

# # Re-identification thread that now searches across the entire frame
# def re_identification_thread():
#     global rectangle, reidentifying, dlib_tracker, kcf_tracker, tracking_initialized, initial_hist, initial_keypoints, initial_descriptors, using_kcf, kcf_start_time, last_bbox_size
#     print("Re-identification thread started.")
    
#     frame_width, frame_height = DESIRED_WIDTH, DESIRED_HEIGHT
#     while True:
#         with frame_lock:
#             current_frame = frame.copy()

#         if orb_active and initial_descriptors is not None:
#             sliding_window_bboxes = generate_adaptive_sliding_windows(frame_width, frame_height, rectangle)
#             found_bbox = parallel_search(current_frame, sliding_window_bboxes, initial_hist, initial_keypoints, initial_descriptors, bf)
#             if found_bbox:
#                 print("Re-identification successful. Switching to KCF tracker.")

#                 # Get the position (x, y) of the newly found bounding box
#                 new_x, new_y, _, _ = found_bbox
                
#                 # Use the last known bounding box size if available
#                 if last_bbox_size is not None:
#                     last_width, last_height = last_bbox_size
#                     found_bbox = (new_x, new_y, last_width, last_height)
#                     print(f"Using last known bounding box size: {last_bbox_size}")
                
#                 # Initialize the KCF tracker with the updated bounding box
#                 kcf_tracker = cv2.TrackerKCF_create()
#                 kcf_tracker.init(current_frame, tuple(found_bbox))
                
#                 # Update histogram and ORB features
#                 initial_hist = compute_color_histogram(current_frame, found_bbox)
#                 initial_keypoints, initial_descriptors = compute_ORB_features(current_frame, found_bbox)
                
#                 tracking_initialized = True

#                 with lock:
#                     reidentifying = False
#                     using_kcf = True
#                     kcf_start_time = time.time()  # Start KCF tracking time
#                 break
#             else:
#                 pass
#         time.sleep(0.1)
#     print("Re-identification thread stopped.")


# Function to change color after a delay
def change_color_after_delay():
    global color, last_command_time, lock, orb_active, object_keypoints, object_descriptors, initial_hist, initial_keypoints, initial_descriptors
    while True:
        time.sleep(1)
        with lock:
            if last_command_time and (time.time() - last_command_time > 5):
                color = (0, 255, 0)  # Change color to green after 5 seconds
                if not orb_active and rectangle is not None:
                    x, y, w, h = rectangle
                    # Make sure frame is available globally
                    initial_keypoints, initial_descriptors = compute_ORB_features(frame, rectangle)
                    initial_hist = compute_color_histogram(frame, rectangle)
                    orb_active = True
                    print("ORB feature extraction activated.")

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
            if command.startswith('p'):
                values = command.split(',')
                x_percent = float(values[0][1:])
                y_percent = float(values[1])
                width_percent = float(values[5])
                height_percent = float(values[6])

                new_rectangle = get_pixel_coordinates(x_percent, y_percent, width_percent, height_percent)

                if new_rectangle[2] > 0 and new_rectangle[3] > 0:
                    rectangle = new_rectangle
                    print(f"Rectangle coordinates: {rectangle}")
                else:
                    print("Invalid rectangle dimensions received. Skipping.")
                    return
                
                current_time = time.time()
                
                with lock:
                    center_x = new_rectangle[0] + new_rectangle[2] // 2
                    center_y = new_rectangle[1] + new_rectangle[3] // 2

                    if tracking_initialized and is_point_inside_rectangle(center_x, center_y, rectangle):
                        if last_command_time and (current_time - last_command_time <= 5):
                            rectangle = new_rectangle
                            print("Updating target within the bounding box.")
                        else:
                            print("New command overlaps with the current tracker after 5 seconds, stopping tracker.")
                            tracking_initialized = False
                            rectangle = None
                            tracker = None
                            color = (0, 255, 0)
                    else:
                        tracker = dlib.correlation_tracker()
                        tracking_initialized = False
                        color = (255, 0, 0)
                        print("New target initialized.")
                    last_command_time = current_time

            else: 
                print(f"Ignoring command: {command}")
    except Exception as e:
        print(f"Error processing message: {e}")

# Video feed thread to process the video and tracking
def video_feed_thread():
    global rectangle, dlib_tracker, kcf_tracker, tracking_initialized, color, frame, initial_keypoints, initial_descriptors, orb_active, lost, initial_hist, reidentifying, reid_thread, using_kcf, kcf_start_time, last_bbox_size, dlib_failure_count

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

    while True:
        ret, frame = cap.read()
        if ret:
            current_time = time.time()
            fps = 1 / (current_time - prev_time)
            prev_time = current_time

            with frame_lock:
                if rectangle is not None:
                    if not tracking_initialized:
                        # Initial tracking by dlib
                        x, y, w, h = rectangle
                        dlib_tracker = dlib.correlation_tracker()  # Initialize dlib tracker
                        dlib_rect = dlib.rectangle(x, y, x + w, y + h)
                        dlib_tracker.start_track(frame, dlib_rect)
                        tracking_initialized = True
                        
                        # Store the last known bounding box size
                        last_bbox_size = (w, h)

                    # Handle KCF tracking after dlib fails
                    if using_kcf:
                        success, bbox = kcf_tracker.update(frame)
                        if success:
                            x, y, w, h = [int(v) for v in bbox]
                            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 3)  # Red box for KCF
                            
                            # Update the last known bounding box size
                            last_bbox_size = (w, h)

                            # Switch back to dlib after 2 seconds of KCF tracking
                            if time.time() - kcf_start_time > 2:
                                print("Switching back to dlib tracker after 2 seconds.")
                                dlib_tracker = dlib.correlation_tracker()  # Reinitialize the dlib tracker
                                dlib_tracker.start_track(frame, dlib.rectangle(x, y, x + w, y + h))
                                using_kcf = False  # Switch back to dlib
                        else:
                            print("KCF tracker lost track. Initiating re-identification.")
                            if not reidentifying:
                                reidentifying = True
                                reid_thread = threading.Thread(target=re_identification_thread, daemon=True)
                                reid_thread.start()

                    # Handle dlib tracking, but only if KCF isn't running
                    if not using_kcf and dlib_tracker is not None:  # Check if dlib_tracker is initialized
                        confidence = dlib_tracker.update(frame)
                        if confidence > 7:
                            pos = dlib_tracker.get_position()
                            x = int(pos.left())
                            y = int(pos.top())
                            width = int(pos.width())
                            height = int(pos.height())
                            cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 3)  # Green box for dlib
                            
                            # Update the last known bounding box size
                            last_bbox_size = (width, height)
                            dlib_failure_count = 0  # Reset the failure count if tracking is successful
                        else:
                            dlib_failure_count += 1
                            print(f"Dlib tracker failed {dlib_failure_count} times.")
                            if dlib_failure_count >= 1:  # Stop dlib, switch to KCF and re-identification
                                print("Dlib tracker failed. Switching to KCF tracker and starting re-identification.")
                                if not reidentifying:
                                    reidentifying = True
                                    reid_thread = threading.Thread(target=re_identification_thread, daemon=True)
                                    reid_thread.start()
                                
                                # Stop the dlib tracker
                                dlib_tracker = None  # Set to None to stop further updates
                                dlib_failure_count = 0  # Reset the failure count after switching to KCF

            cv2.putText(frame, f"FPS: {fps:.2f}", (100, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            if using_kcf:
                cv2.putText(frame, "KCF Tracker", (100, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            else:
                cv2.putText(frame, "Dlib Correlation Tracker", (100, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            out.write(frame)
        if cv2.waitKey(33) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()


# Re-identification thread that now searches across the entire frame and switches to KCF
def re_identification_thread():
    global rectangle, reidentifying, kcf_tracker, tracking_initialized, initial_hist, initial_keypoints, initial_descriptors, using_kcf, kcf_start_time, last_bbox_size
    print("Re-identification thread started.")

    frame_width, frame_height = DESIRED_WIDTH, DESIRED_HEIGHT
    while True:
        with frame_lock:
            if frame is None:
                continue
            current_frame = frame.copy()

        if orb_active and initial_descriptors is not None:
            sliding_window_bboxes = generate_adaptive_sliding_windows(frame_width, frame_height, rectangle)
            found_bbox = parallel_search(current_frame, sliding_window_bboxes, initial_hist, initial_keypoints, initial_descriptors, bf)
            if found_bbox:
                print("Re-identification successful. Switching to KCF tracker.")

                # Get the position (x, y) of the newly found bounding box
                new_x, new_y, _, _ = found_bbox

                # Use the last known bounding box size if available
                if last_bbox_size is not None:
                    last_width, last_height = last_bbox_size
                    found_bbox = (new_x, new_y, last_width, last_height)
                    print(f"Using last known bounding box size: {last_bbox_size}")

                # Initialize the KCF tracker with the updated bounding box
                print("Initializing KCF tracker...")
                kcf_tracker = cv2.TrackerKCF_create()
                kcf_initialized = kcf_tracker.init(current_frame, tuple(found_bbox))

                if kcf_initialized:
                    print("KCF tracker initialized successfully.")
                    # Update histogram and ORB features
                    initial_hist = compute_color_histogram(current_frame, found_bbox)
                    initial_keypoints, initial_descriptors = compute_ORB_features(current_frame, found_bbox)

                    tracking_initialized = True

                    with lock:
                        reidentifying = False
                        using_kcf = True
                        kcf_start_time = time.time()  # Start KCF tracking time
                    break
                else:
                    print("KCF tracker initialization failed.")
        time.sleep(0.1)
    print("Re-identification thread stopped.")


# Initialize the rest of your system
threading.Thread(target=change_color_after_delay, daemon=True).start()

client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message
client.connect("192.168.1.61", 1883, 60)

threading.Thread(target=video_feed_thread, daemon=True).start()
client.loop_forever()
