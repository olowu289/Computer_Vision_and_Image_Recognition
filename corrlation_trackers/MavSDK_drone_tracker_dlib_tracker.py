import cv2
import threading
import paho.mqtt.client as mqtt
import time
import numpy as np
import concurrent.futures
import dlib
import asyncio
from mavsdk import System
from mavsdk.offboard import VelocityNedYaw

# MAVSDK Drone System
drone = System()

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
drone_armed_airborne = False  # Tracks if the drone is armed and airborne

# Default bounding box size
DEFAULT_WIDTH = 150
DEFAULT_HEIGHT = 200

# Target distance (default 1 meter from the target)
TARGET_DISTANCE = 1.0

# GStreamer pipeline configuration
GSTREAMER_PIPELINE = (
    'appsrc ! videoconvert ! x264enc speed-preset=ultrafast tune=zerolatency ! '
    'rtph264pay config-interval=1 pt=96 ! udpsink host=192.168.1.61 port=5600'
)

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)  # ORB matcher


# MAVSDK Drone Functions
async def connect_to_drone():
    print("Connecting to the drone...")
    await drone.connect(system_address="udp://:14540")
    async for state in drone.core.connection_state():
        if state.is_connected:
            print("Drone connected!")
            return True
    return False


async def initialize_drone():
    print("Setting takeoff altitude...")
    await drone.action.set_takeoff_altitude(5.0)
    await drone.action.arm()
    print("Drone armed!")


async def takeoff_to_altitude(altitude):
    global drone_armed_airborne
    print(f"Taking off to altitude: {altitude} meters...")
    await drone.action.arm()
    await drone.action.takeoff()
    while True:
        async for position in drone.telemetry.position():
            if position.relative_altitude_m >= altitude - 0.1:  # Allow for slight tolerance
                print(f"Reached altitude: {altitude} meters.")
                drone_armed_airborne = True
                return
        await asyncio.sleep(0.1)


async def send_ned_velocity(north, east, down, duration):
    print("Sending NED velocity commands...")
    try:
        await drone.offboard.start()
        for _ in range(duration):
            velocity = VelocityNedYaw(north_m_s=north, east_m_s=east, down_m_s=down, yaw_deg=0.0)
            await drone.offboard.set_velocity_ned(velocity)
            await asyncio.sleep(1)
    except Exception as e:
        print(f"Error while sending NED velocity: {e}")
    finally:
        await drone.offboard.stop()
    print("NED velocity commands sent.")


async def drone_status_monitor():
    global drone_armed_airborne
    while True:
        armed = False
        async for state in drone.telemetry.armed():
            armed = state
            break
        airborne = await is_drone_airborne()
        drone_armed_airborne = armed and airborne
        print(f"Drone armed: {armed}, Airborne: {airborne}, Status: {drone_armed_airborne}")
        await asyncio.sleep(1)


async def is_drone_airborne():
    async for position in drone.telemetry.position():
        return position.relative_altitude_m > 0.5


def calculate_offsets_from_frame_center(frame_width, frame_height, target_center):
    frame_center_x = frame_width // 2
    frame_center_y = frame_height // 2
    offset_x = target_center[0] - frame_center_x
    offset_y = target_center[1] - frame_center_y
    return offset_x, offset_y


def calculate_ned_velocity(offset_x, offset_y, current_distance, scale_factor=0.02):
    north_velocity = -offset_y * scale_factor  # Adjust north-south based on vertical offset
    east_velocity = offset_x * scale_factor    # Adjust east-west based on horizontal offset

    # Adjust altitude to maintain target distance
    down_velocity = 0.0
    if current_distance < TARGET_DISTANCE - 0.1:
        down_velocity = -0.1  # Ascend to increase distance
    elif current_distance > TARGET_DISTANCE + 0.1:
        down_velocity = 0.1   # Descend to decrease distance

    return north_velocity, east_velocity, down_velocity


def video_feed_thread(loop):
    global rectangle, tracker, tracking_initialized, color, frame

    cap = cv2.VideoCapture("/dev/video0")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, DESIRED_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DESIRED_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, DESIRED_FPS)

    cv2.namedWindow("Video Feed")
    cv2.setMouseCallback("Video Feed", mouse_click_event)

    while True:
        ret, frame = cap.read()
        if ret:
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
                        x, y, w, h = int(pos.left()), int(pos.top()), int(pos.width()), int(pos.height())
                        target_center = (x + w // 2, y + h // 2)

                        offset_x, offset_y = calculate_offsets_from_frame_center(DESIRED_WIDTH, DESIRED_HEIGHT, target_center)
                        current_distance = TARGET_DISTANCE  # Replace with actual telemetry if available

                        north_velocity, east_velocity, down_velocity = calculate_ned_velocity(offset_x, offset_y, current_distance)

                        asyncio.run_coroutine_threadsafe(
                            send_ned_velocity(north_velocity, east_velocity, down_velocity, 1),
                            loop,
                        )

                        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)
                        cv2.circle(frame, target_center, 5, (0, 255, 0), -1)

            cv2.imshow("Video Feed", frame)

        if cv2.waitKey(33) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def mouse_click_event(event, x, y, flags, param):
    global rectangle, tracker, tracking_initialized, color
    if event == cv2.EVENT_LBUTTONDOWN:
        with lock:
            rectangle = (x - 75, y - 100, 150, 200)
            tracker = dlib.correlation_tracker()
            tracker.start_track(frame, dlib.rectangle(*rectangle))
            tracking_initialized = True
            color = (255, 0, 0)
            print(f"Mouse clicked at: ({x}, {y}), Rectangle: {rectangle}")


async def main():
    if await connect_to_drone():
        asyncio.create_task(drone_status_monitor())

        loop = asyncio.get_event_loop()
        video_thread = threading.Thread(target=video_feed_thread, args=(loop,), daemon=True)
        video_thread.start()

        await takeoff_to_altitude(1.0)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Program interrupted.")
