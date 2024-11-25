import cv2
import threading
import paho.mqtt.client as mqtt
import time

# Desired video properties
DESIRED_WIDTH = 1920
DESIRED_HEIGHT = 1080
DESIRED_FPS = 30

# Global variables for a single target
rectangle = None
tracker = None
tracking_initialized = False
last_command_time = None
color = (255, 0, 0)  # Initial color (red)
lock = threading.Lock()

# GStreamer pipeline configuration
GSTREAMER_PIPELINE = (
    'appsrc ! videoconvert ! x264enc speed-preset=ultrafast tune=zerolatency ! '
    'rtph264pay config-interval=1 pt=96 ! udpsink host=192.168.1.12 port=5600'
)

# # Fixed screen size
# FIXED_WIDTH = DESIRED_WIDTH
# FIXED_HEIGHT = DESIRED_HEIGHT

def get_pixel_coordinates(x_percent, y_percent, width_percent, height_percent):
    """
    Convert percentage values to pixel coordinates based on the fixed screen resolution.
    """
    center_x = int((x_percent / 100) * DESIRED_WIDTH)
    center_y = int((y_percent / 100) *DESIRED_HEIGHT)
    width_pixel = int(width_percent * DESIRED_WIDTH)  # No division by 100 for width
    height_pixel = int(height_percent * DESIRED_HEIGHT)  # No division by 100 for height
    x_pixel = center_x
    y_pixel = center_y
    print("x_pixel: ", x_pixel, " y_pixel: ", y_pixel, " width_pixel: ", width_pixel, " height_pixel: ", height_pixel)
    print("center x: ", center_x, " center_y: ", center_y)
    print("x_percent: ", x_percent, "y_percent: ", y_percent,  "width_percent: ", width_percent , "height_percent: ", height_percent)
    return x_pixel, y_pixel, width_pixel, height_pixel


def is_point_inside_rectangle(x, y, rect):
    """Check if a point (x, y) is inside a given rectangle."""
    rx, ry, rw, rh = rect
    return rx <= x <= rx + rw and ry <= y <= ry + rh

def change_color_after_delay():
    global color, last_command_time, lock
    
    while True:
        time.sleep(1)  # Check every 1 second
        with lock:
            if last_command_time and (time.time() - last_command_time > 5):
                color = (0, 255, 0)  # Change to green after 5 seconds

def on_connect(client, userdata, flags, rc):
    print(f"Connected with result code {rc}")
    client.subscribe("drone/com")

def on_message(client, userdata, msg):
    global rectangle, tracker, tracking_initialized, last_command_time, color, lock
    
    try:
        # Decode the message payload
        payload = msg.payload.decode()

        # Split the payload into individual commands using newline as a separator
        commands = payload.split('\n')
        print(commands)

        for command in commands:
            # Strip any leading or trailing whitespace/newlines
            command = command.strip()

            if not command:
                continue  # Skip empty commands

            # Check if the command starts with 'p'
            if command.startswith('p'):
                # Split the command by commas to get individual values
                values = command.split(',')

                # Extract the x and y percentages by removing the 'p' from the first value
                x_percent = float(values[0][1:])
                y_percent = float(values[1])

                # Extract the width and height percentages
                width_percent = float(values[5])
                height_percent = float(values[6])

                print(f"Received valid data: x={x_percent}, y={y_percent}, width={width_percent}, height={height_percent}")

                # Convert percentages to pixel coordinates relative to the fixed screen size
                new_rectangle = get_pixel_coordinates(x_percent, y_percent, width_percent, height_percent)
                
                # Ensure the rectangle is valid
                if new_rectangle[2] > 0 and new_rectangle[3] > 0:
                    rectangle = new_rectangle
                    # Print the datatype of rectangle coordinates
                    print(f"Rectangle coordinates: {rectangle}")
                else:
                    print("Invalid rectangle dimensions received. Skipping.")
                    return

                # Get current time
                current_time = time.time()

                with lock:
                    # Check if the center of the new rectangle is inside the existing active rectangle
                    center_x = new_rectangle[0] + new_rectangle[2] // 2
                    center_y = new_rectangle[1] + new_rectangle[3] // 2

                    if tracking_initialized and is_point_inside_rectangle(center_x, center_y, rectangle):
                        # If the new command comes within 5 seconds, update the target without stopping
                        if last_command_time and (current_time - last_command_time <= 5):
                            rectangle = new_rectangle
                            print("Updating target within the bounding box.")
                        else:
                            print("New command overlaps with the current tracker after 5 seconds, stopping tracker.")
                            tracking_initialized = False
                            rectangle = None
                            tracker = None
                            color = (0, 255, 0)  # Change to green after 5 seconds
                    else:
                        # Initialize a new target
                        tracker = cv2.TrackerCSRT_create()
                        tracking_initialized = False
                        color = (255, 0, 0)  # Red for a new target
                        print("New target initialized.")

                    last_command_time = current_time

            else:
                print(f"Ignoring command: {command}")
            
    except Exception as e:
        print(f"Error processing message: {e}")

def video_feed_thread():
    global rectangle, tracker, tracking_initialized, color, frame

    # Initialize video capture from the webcam
    cap = cv2.VideoCapture("/dev/video0")
    
    # Set frame resolution and fps
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, DESIRED_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DESIRED_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, DESIRED_FPS)

    # Create a GStreamer pipeline for streaming
    out = cv2.VideoWriter(GSTREAMER_PIPELINE, cv2.CAP_GSTREAMER, 0, DESIRED_FPS, (DESIRED_WIDTH, DESIRED_HEIGHT), True)

    if not out.isOpened():
        print("Failed to open GStreamer pipeline.")
        return

    while True:
        # Capture the video frame by frame
        ret, frame = cap.read()

        if ret:
            if rectangle is not None:
                if not tracking_initialized:
                    print(f"Initializing tracker with rectangle: {rectangle}")
                    tracker.init(frame, tuple(rectangle))
                    tracking_initialized = True

                # Update the tracker and get the updated position of the rectangle
                success, rect = tracker.update(frame)
                if success:
                    # Draw the updated rectangle on the frame
                    x, y, width, height = map(int, rect)
                    cv2.rectangle(frame, (x, y), (x + width, y + height), color, 3)
                    center_x = x + width // 2
                    center_y = y + height // 2
                    cv2.circle(frame, (center_x, center_y), 1, (0, 255, 0), 1)  # Draw a small circle at the center
                else:
                    print("Tracking failed for the current target. Removing bounding box.")
                    tracking_initialized = False
                    rectangle = None  # Clear the rectangle to remove the bounding box

            # Write the frame to the GStreamer pipeline
            out.write(frame)

        # Wait for a short time (33 ms) to simulate video feed frame rate (~30 FPS)
        if cv2.waitKey(33) & 0xFF == ord('q'):
            break

    # Release the capture and the GStreamer pipeline
    cap.release()
    out.release()

# Start the color change thread
threading.Thread(target=change_color_after_delay, daemon=True).start()

# MQTT client setup
client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message

# Replace with your MQTT broker's IP address
client.connect("192.168.1.12", 1883, 60)

# Start the video feed thread
threading.Thread(target=video_feed_thread, daemon=True).start()

# Start the MQTT loop to process incoming messages
client.loop_forever()
