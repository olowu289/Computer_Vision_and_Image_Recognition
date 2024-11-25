import cv2
import time

# Open the webcam
cap = cv2.VideoCapture(0)  # You can replace 0 with the appropriate camera index or video file

if not cap.isOpened():
    print("Error: Could not open video source.")
    exit()

cv2.namedWindow("Video Stream")

# GStreamer pipeline for streaming to QGroundControl
pipeline_str = (
    "v4l2src ! "
    "video/x-raw,width=640,height=480 ! "
    "videoconvert ! "
    "x264enc tune=zerolatency ! "
    "rtph264pay config-interval=1 pt=96 ! "
    "udpsink host=192.168.1.61 port=5600"  # Replace with QGroundControl's IP address
)

out = cv2.VideoWriter(pipeline_str, cv2.CAP_GSTREAMER, 0, 30, (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

if not out.isOpened():
    print("Error: Could not open video writer.")
    cap.release()
    exit()

fps = 0
start_time = time.time()

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # Calculate and display FPS
        end_time = time.time()
        fps = 1 / (end_time - start_time)
        start_time = end_time
        cv2.putText(frame, f'FPS: {int(fps)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Video Stream", frame)
        out.write(frame)  # Write the frame to the GStreamer pipeline

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
except KeyboardInterrupt:
    pass

# Cleanup
cap.release()
out.release()
cv2.destroyAllWindows()
