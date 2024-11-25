import cv2
import numpy as np

# Function to compute color histogram
def compute_color_histogram(image, bbox):
    x, y, w, h = [int(v) for v in bbox]
    # Ensure the bounding box is within the image boundaries
    if w == 0 or h == 0 or x < 0 or y < 0 or x + w > image.shape[1] or y + h > image.shape[0]:
        print("Invalid bounding box.")
        return None
    
    roi = image[y:y+h, x:x+w]

    if roi.size == 0:
        print("Empty region of interest (ROI).")
        return None

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
    return hist

# Function to compute ORB keypoints and descriptors
def compute_ORB_features(image, bbox):
    x, y, w, h = [int(v) for v in bbox]
    roi = image[y:y+h, x:x+w]
    orb = cv2.ORB_create()
    keypoints, descriptors = orb.detectAndCompute(roi, None)
    return keypoints, descriptors

# Function to compare histograms
def compare_histograms(hist1, hist2):
    return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)


# Initialize video capture from the webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Could not open webcam")
    exit()

# Read the first frame from the webcam
ret, frame = cap.read()
if not ret:
    print("Failed to read from webcam")
    exit()

# Let the user select a region of interest (ROI) for tracking
print("Select a region of interest (ROI) to track.")
bbox = cv2.selectROI("Select ROI", frame, fromCenter=False, showCrosshair=True)

# Initialize the CSRT tracker with the selected ROI
tracker = cv2.TrackerCSRT_create()
tracker.init(frame, bbox)

# Compute the initial color histogram and ORB features for re-identification
initial_hist = compute_color_histogram(frame, bbox)
initial_keypoints, initial_descriptors = compute_ORB_features(frame, bbox)

# BFMatcher for feature matching
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

lost = False
cv2.destroyWindow("Select ROI")  # Close the ROI selection window

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Update the tracker
    ret, bbox = tracker.update(frame)

    if ret:
        # Object is being tracked
        if lost:
            print("Tracking resumed.")
        lost = False

        # Draw the bounding box
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)

        # Compute color histogram and ORB features for the current frame
        current_hist = compute_color_histogram(frame, bbox)
        current_keypoints, current_descriptors = compute_ORB_features(frame, bbox)

        if current_hist is None:
            pass
            # print("Invalid bounding box, skipping frame.")
    else:
        # Object lost, mark as lost
        if not lost:
            print("Object lost, trying to re-identify...")
        lost = True

    if lost:
        # Try to re-identify object in the frame using both color histogram and ORB features
        for i in range(0, frame.shape[0], 50):
            for j in range(0, frame.shape[1], 50):
                # Create a small bounding box around each grid point
                search_bbox = (j, i, 150, 150)  # Use a fixed size for searching
                search_hist = compute_color_histogram(frame, search_bbox)
                search_keypoints, search_descriptors = compute_ORB_features(frame, search_bbox)

                if search_hist is None or search_descriptors is None:
                    continue

                # Compare color histograms
                similarity = compare_histograms(initial_hist, search_hist)
                if similarity > 0.8:
                    # Match ORB features using BFMatcher
                    matches = bf.match(initial_descriptors, search_descriptors)
                    if len(matches) > 15:  # Use a threshold for good matches
                        print("Object re-identified.")
                        tracker.init(frame, search_bbox)  # Re-initialize the tracker
                        lost = False
                        break
            if not lost:
                break

    # Display the tracking result
    cv2.imshow("Tracking", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()