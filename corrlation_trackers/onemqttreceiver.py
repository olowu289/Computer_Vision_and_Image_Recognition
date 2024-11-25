import cv2
import threading
import paho.mqtt.client as mqtt
import time

# Global variables for two targets
rectangles = [None, None]
trackers = [None, None]
tracking_initialized = [False, False]
last_command_time = [None, None]
colors = [(255, 0, 0), (0, 0, 255)]  # Initial colors (red, blue)
lock = threading.Lock()

# GStreamer pipeline configuration
GSTREAMER_PIPELINE = (
    'appsrc ! videoconvert ! x264enc speed-preset=ultrafast tune=zerolatency ! rtph264pay config-interval=1 pt=96 ! udpsink host=192.168.1.12 port=5600'
)

def get_pixel_coordinates(x_percent, y_percent, width_percent, height_percent, frame_width, frame_height):
    """
    Convert percentage values to pixel coordinates based on the current video feed resolution.
    """
    center_x = int((x_percent / 100) * frame_width)
    center_y = int((y_percent / 100) * frame_height)
    width_pixel = int((width_percent / 100) * frame_width)
    height_pixel = int((height_percent / 100) * frame_height)
    x_pixel = center_x - width_pixel // 2
    y_pixel = center_y - height_pixel // 2
    
    return x_pixel, y_pixel, width_pixel, height_pixel

def is_point_inside_rectangle(x, y, rect):
    """Check if a point (x, y) is inside a given rectangle."""
    rx, ry, rw, rh = rect
    return rx <= x <= rx + rw and ry <= y <= ry + rh

def change_color_after_delay():
    global colors, last_command_time, lock
    
    while True:
        time.sleep(1)
        with lock:
            for i in range(2):
                if last_command_time[i] and (time.time() - last_command_time[i] > 5):
                    colors[i] = (0, 255, 0)  # Change to green after 5 seconds

def on_connect(client, userdata, flags, rc):
    print(f"Connected with result code {rc}")
    client.subscribe("drone/com")

def on_message(client, userdata, msg):
    global rectangles, trackers, tracking_initialized, last_command_time, colors, lock
    
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

                frame_height, frame_width = frame.shape[:2]
                new_rectangle = get_pixel_coordinates(x_percent, y_percent, width_percent, height_percent, frame_width, frame_height)
                current_time = time.time()

                with lock:
                    center_x = new_rectangle[0] + new_rectangle[2] // 2
                    center_y = new_rectangle[1] + new_rectangle[3] // 2

                    # Check if there is an available slot or overlap
                    target_added = False
                    for i in range(2):
                        if tracking_initialized[i] and is_point_inside_rectangle(center_x, center_y, rectangles[i]):
                            if last_command_time[i] and (current_time - last_command_time[i] <= 5):
                                rectangles[i] = new_rectangle
                                print(f"Updating target {i + 1} within the bounding box.")
                            else:
                                print(f"Stopping target {i + 1} after 5 seconds.")
                                tracking_initialized[i] = False
                                rectangles[i] = None
                                trackers[i] = None
                                colors[i] = (0, 255, 0)
                            last_command_time[i] = current_time
                            target_added = True
                            break
                    
                    # Add new target if there's an available slot
                    if not target_added:
                        for i in range(2):
                            if not tracking_initialized[i]:
                                rectangles[i] = new_rectangle
                                trackers[i] = cv2.TrackerCSRT_create()
                                tracking_initialized[i] = False
                                colors[i] = (255, 0, 0)
                                last_command_time[i] = current_time
                                print(f"New target {i + 1} initialized.")
                                target_added = True
                                break

                    # If both slots are full, replace the oldest target
                    if not target_added:
                        oldest_index = 0 if last_command_time[0] < last_command_time[1] else 1
                        rectangles[oldest_index] = new_rectangle
                        trackers[oldest_index] = cv2.TrackerCSRT_create()
                        tracking_initialized[oldest_index] = False
                        colors[oldest_index] = (255, 0, 0)
                        last_command_time[oldest_index] = current_time
                        print(f"Replaced target {oldest_index + 1} with a new target.")
            
            else:
                print(f"Ignoring command: {command}")
            
    except Exception as e:
        print(f"Error processing message: {e}")

def video_feed_thread():
    global rectangles, trackers, tracking_initialized, colors, frame

    cap = cv2.VideoCapture("/dev/video0")
    out = cv2.VideoWriter(GSTREAMER_PIPELINE, cv2.CAP_GSTREAMER, 0, 30, (int(cap.get(3)), int(cap.get(4))), True)

    if not out.isOpened():
        print("Failed to open GStreamer pipeline.")
        return

    while True:
        ret, frame = cap.read()

        if ret:
            for i in range(2):
                if rectangles[i] is not None:
                    if not tracking_initialized[i]:
                        trackers[i].init(frame, tuple(rectangles[i]))
                        tracking_initialized[i] = True

                    success, rect = trackers[i].update(frame)
                    if success:
                        x, y, width, height = map(int, rect)
                        cv2.rectangle(frame, (x, y), (x + width, y + height), colors[i], 3)
                    else:
                        print(f"Tracking failed for target {i + 1}.")
                        tracking_initialized[i] = False
                        rectangles[i] = None

            out.write(frame)

        if cv2.waitKey(33) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()

threading.Thread(target=change_color_after_delay, daemon=True).start()

client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message

client.connect("192.168.1.12", 1883, 60)
threading.Thread(target=video_feed_thread, daemon=True).start()
client.loop_forever()
