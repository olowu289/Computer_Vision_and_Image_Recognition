#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <dlib/opencv.h>
#include <dlib/image_processing.h>
#include <dlib/image_io.h>
#include <iostream>
#include <thread>
#include <mutex>
#include <chrono>
#include <condition_variable>
#include <mosquitto.h>
#include <vector>

// Desired video properties
const int DESIRED_WIDTH = 1920;
const int DESIRED_HEIGHT = 1080;
const int DESIRED_FPS = 30;

// Global variables for a single target
cv::Rect rectangle;
dlib::correlation_tracker tracker;
bool tracking_initialized = false;
time_t last_command_time;
cv::Scalar color(255, 0, 0);  // Initial color (red)
std::mutex lock;
std::vector<cv::KeyPoint> object_keypoints;
cv::Mat object_descriptors;
cv::Mat initial_hist;
bool orb_active = false;
bool lost = false;
bool reidentifying = false;  // Flag to track if re-identification is running
std::thread reid_thread;     // Re-identification thread
std::mutex frame_lock;       // Lock for the frame

// GStreamer pipeline configuration
std::string GSTREAMER_PIPELINE = "appsrc ! videoconvert ! x264enc speed-preset=ultrafast tune=zerolatency ! "
                                 "rtph264pay config-interval=1 pt=96 ! udpsink host=192.168.1.61 port=5600";

cv::BFMatcher bf(cv::NORM_HAMMING, false);  // ORB matcher

// Function to convert percentage to pixel coordinates based on screen resolution
cv::Rect get_pixel_coordinates(float x_percent, float y_percent, float width_percent, float height_percent, int feed_width = DESIRED_WIDTH, int feed_height = DESIRED_HEIGHT) {
    int center_x = static_cast<int>((x_percent / 100.0) * feed_width);
    int center_y = static_cast<int>((y_percent / 100.0) * feed_height);
    int width_pixel = static_cast<int>((width_percent / 100.0) * feed_width);
    int height_pixel = static_cast<int>((height_percent / 100.0) * feed_height);
    int x_pixel = center_x - width_pixel / 2;
    int y_pixel = center_y - height_pixel / 2;
    return cv::Rect(x_pixel, y_pixel, width_pixel, height_pixel);
}

// Function to check if a point is inside a rectangle
bool is_point_inside_rectangle(int x, int y, const cv::Rect& rect) {
    return (rect.x <= x && x <= rect.x + rect.width && rect.y <= y && y <= rect.y + rect.height);
}

// Function to extract high pixel intensity region within the bounding box
std::tuple<cv::Mat, cv::Mat, std::vector<uchar>> extract_high_pixel_region(const cv::Mat& frame, const cv::Rect& bbox, int threshold = 100) {
    if (bbox.x < 0 || bbox.y < 0 || bbox.x + bbox.width > frame.cols || bbox.y + bbox.height > frame.rows) {
        return {cv::Mat(), cv::Mat(), std::vector<uchar>()};
    }
    cv::Mat roi = frame(bbox);
    cv::Mat gray_roi;
    cv::cvtColor(roi, gray_roi, cv::COLOR_BGR2GRAY);
    cv::Mat high_pixel_mask;
    cv::threshold(gray_roi, high_pixel_mask, threshold, 255, cv::THRESH_BINARY_INV);
    std::vector<uchar> high_pixel_values;
    for (int y = 0; y < gray_roi.rows; ++y) {
        for (int x = 0; x < gray_roi.cols; ++x) {
            if (high_pixel_mask.at<uchar>(y, x) == 255) {
                high_pixel_values.push_back(gray_roi.at<uchar>(y, x));
            }
        }
    }
    return {high_pixel_mask, roi, high_pixel_values};
}

// Modified compute_color_histogram to only focus on high pixel intensity region
cv::Mat compute_color_histogram(const cv::Mat& image, const cv::Rect& bbox) {
    auto [high_pixel_mask, roi, _] = extract_high_pixel_region(image, bbox);
    if (high_pixel_mask.empty() || roi.empty()) return cv::Mat();

    cv::Mat hsv, hist;
    cv::cvtColor(roi, hsv, cv::COLOR_BGR2HSV);
    int h_bins = 180, s_bins = 256;
    int hist_size[] = { h_bins, s_bins };
    float h_ranges[] = { 0, 180 };
    float s_ranges[] = { 0, 256 };
    const float* ranges[] = { h_ranges, s_ranges };
    int channels[] = { 0, 1 };
    cv::calcHist(&hsv, 1, channels, high_pixel_mask, hist, 2, hist_size, ranges, true, false);
    cv::normalize(hist, hist, 0, 1, cv::NORM_MINMAX, -1, cv::Mat());
    return hist;
}

// Modified compute_ORB_features to only focus on keypoints in the high pixel intensity region
std::tuple<std::vector<cv::KeyPoint>, cv::Mat> compute_ORB_features(const cv::Mat& image, const cv::Rect& bbox) {
    auto [high_pixel_mask, roi, _] = extract_high_pixel_region(image, bbox);
    if (high_pixel_mask.empty() || roi.empty()) return { std::vector<cv::KeyPoint>(), cv::Mat() };

    cv::Ptr<cv::ORB> orb = cv::ORB::create();
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    orb->detectAndCompute(roi, high_pixel_mask, keypoints, descriptors);
    return { keypoints, descriptors };
}

// Function to compare histograms
double compare_histograms(const cv::Mat& hist1, const cv::Mat& hist2) {
    return cv::compareHist(hist1, hist2, cv::HISTCMP_CORREL);
}

// Function to search the entire frame for re-identification
cv::Rect search_entire_frame(const cv::Mat& frame, const cv::Mat& initial_hist, const std::vector<cv::KeyPoint>& initial_keypoints, const cv::Mat& initial_descriptors, cv::BFMatcher& bf) {
    // Compute color histogram for the entire frame
    cv::Mat frame_hist = compute_color_histogram(frame, cv::Rect(0, 0, frame.cols, frame.rows));  // Full frame
    auto [frame_keypoints, frame_descriptors] = compute_ORB_features(frame, cv::Rect(0, 0, frame.cols, frame.rows));  // Full frame

    // Check if histogram or descriptors are invalid
    if (frame_hist.empty() || frame_descriptors.empty()) return cv::Rect();

    // Compare histograms
    double similarity = compare_histograms(initial_hist, frame_hist);

    // Use a relaxed matching criteria with ORB features
    std::vector<std::vector<cv::DMatch>> matches;
    bf.knnMatch(initial_descriptors, frame_descriptors, matches, 2);
    std::vector<cv::DMatch> good_matches;
    for (auto& match : matches) {
        if (match[0].distance < 0.85 * match[1].distance) {
            good_matches.push_back(match[0]);
        }
    }

    // Reduced similarity threshold and match count for re-identification
    if (similarity > 0.75 && good_matches.size() > 5) {
        return cv::Rect(0, 0, frame.cols, frame.rows);  // Return entire frame as the bounding box
    }

    return cv::Rect();
}

// Sliding window search using parallel processing (basic version, no threads here for simplicity)
cv::Rect parallel_search(const cv::Mat& frame, const std::vector<cv::Rect>& bboxes, const cv::Mat& initial_hist, const std::vector<cv::KeyPoint>& initial_keypoints, const cv::Mat& initial_descriptors, cv::BFMatcher& bf) {
    for (const auto& bbox : bboxes) {
        cv::Rect found_bbox = search_entire_frame(frame, initial_hist, initial_keypoints, initial_descriptors, bf);
        if (found_bbox.area() > 0) {
            return found_bbox;
        }
    }
    return cv::Rect();
}

// Global sliding window generation with larger windows and smaller step size
std::vector<cv::Rect> generate_sliding_windows(int frame_width, int frame_height, int window_size = 100, int step_size = 15) {
    std::vector<cv::Rect> bboxes;
    for (int y = 0; y < frame_height - window_size; y += step_size) {
        for (int x = 0; x < frame_width - window_size; x += step_size) {
            bboxes.push_back(cv::Rect(x, y, window_size, window_size));
        }
    }
    return bboxes;
}

// Re-identification thread that now searches across the entire frame
void re_identification_thread() {
    std::cout << "Re-identification thread started." << std::endl;
    int frame_width = DESIRED_WIDTH;
    int frame_height = DESIRED_HEIGHT;

    while (true) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));

        if (orb_active && !object_descriptors.empty()) {
            std::lock_guard<std::mutex> guard(frame_lock);
            cv::Mat current_frame;  // Assume this frame is obtained from video feed

            auto sliding_window_bboxes = generate_sliding_windows(frame_width, frame_height);
            cv::Rect found_bbox = parallel_search(current_frame, sliding_window_bboxes, initial_hist, object_keypoints, object_descriptors, bf);
            if (found_bbox.area() > 0) {
                std::cout << "Re-identification successful." << std::endl;
                tracker.start_track(current_frame, dlib::rectangle(found_bbox.x, found_bbox.y, found_bbox.x + found_bbox.width, found_bbox.y + found_bbox.height));
                initial_hist = compute_color_histogram(current_frame, found_bbox);  // Update histogram
                auto [new_keypoints, new_descriptors] = compute_ORB_features(current_frame, found_bbox);  // Update ORB features
                object_keypoints = new_keypoints;
                object_descriptors = new_descriptors;
                tracking_initialized = true;

                std::lock_guard<std::mutex> lock_guard(lock);
                reidentifying = false;
                break;  // Exit the loop once re-identification is successful
            }
        }
    }
    std::cout << "Re-identification thread stopped." << std::endl;
}

// MQTT message handling functions
void on_connect(struct mosquitto* client, void* userdata, int rc) {
    std::cout << "Connected with result code " << rc << std::endl;
    mosquitto_subscribe(client, NULL, "drone/com", 0);
}

void on_message(struct mosquitto* client, void* userdata, const struct mosquitto_message* msg) {
    std::lock_guard<std::mutex> guard(lock);
    std::string payload(static_cast<char*>(msg->payload), msg->payloadlen);
    std::cout << "Message received: " << payload << std::endl;

    // Process message to extract rectangle coordinates (assuming format)
    if (payload[0] == 'p') {
        // Extract percentages from the message payload and update the rectangle
        // Assuming a fixed format "pX%,Y%,W%,H%"
        float x_percent = std::stof(payload.substr(1, 2));
        float y_percent = std::stof(payload.substr(4, 2));
        float width_percent = std::stof(payload.substr(7, 2));
        float height_percent = std::stof(payload.substr(10, 2));

        rectangle = get_pixel_coordinates(x_percent, y_percent, width_percent, height_percent);
        std::cout << "Updated rectangle: " << rectangle << std::endl;
    }
}

// Video feed thread to process the video and tracking
void video_feed_thread() {
    cv::VideoCapture cap(0);  // Open default camera
    cap.set(cv::CAP_PROP_FRAME_WIDTH, DESIRED_WIDTH);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, DESIRED_HEIGHT);
    cap.set(cv::CAP_PROP_FPS, DESIRED_FPS);

    cv::VideoWriter out(GSTREAMER_PIPELINE, cv::CAP_GSTREAMER, 0, DESIRED_FPS, cv::Size(DESIRED_WIDTH, DESIRED_HEIGHT), true);
    if (!out.isOpened()) {
        std::cerr << "Failed to open GStreamer pipeline." << std::endl;
        return;
    }

    while (true) {
        cv::Mat frame;
        cap >> frame;  // Capture a new frame

        if (!frame.empty()) {
            std::lock_guard<std::mutex> guard(frame_lock);

            // Update tracker if initialized
            if (tracking_initialized) {
                tracker.update(frame);  // Update tracker with the new frame
                dlib::rectangle dlib_rect = tracker.get_position();
                cv::Rect tracked_rect(dlib_rect.left(), dlib_rect.top(), dlib_rect.width(), dlib_rect.height());
                cv::rectangle(frame, tracked_rect, color, 3);
            }

            out.write(frame);  // Write the frame to GStreamer pipeline
        }

        if (cv::waitKey(33) == 'q') break;  // Break on 'q' key press
    }

    cap.release();
    out.release();
}

// Main function to initialize system and start threads
int main() {
    mosquitto_lib_init();
    struct mosquitto* client = mosquitto_new(NULL, true, NULL);
    mosquitto_connect_callback_set(client, on_connect);
    mosquitto_message_callback_set(client, on_message);

    if (mosquitto_connect(client, "192.168.1.61", 1883, 60) != MOSQ_ERR_SUCCESS) {
        std::cerr << "Failed to connect to MQTT broker." << std::endl;
        return -1;
    }

    // Start the video feed thread
    std::thread video_thread(video_feed_thread);
    video_thread.detach();

    // Start the re-identification thread
    reid_thread = std::thread(re_identification_thread);
    reid_thread.detach();

    // Start MQTT loop
    mosquitto_loop_start(client);

    // Run indefinitely
    std::this_thread::sleep_for(std::chrono::hours(24));

    mosquitto_loop_stop(client, true);
    mosquitto_destroy(client);
    mosquitto_lib_cleanup();

    return 0;
}
