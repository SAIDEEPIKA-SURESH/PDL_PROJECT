import os
import cv2
import numpy as np
import pyttsx3
import threading
import queue  # For text-to-speech
from lib import glib
import time  # For cooldown timing

# Initialize pyttsx3 for text-to-speech
engine = pyttsx3.init()
engine.setProperty('rate', 150)

# Queue for text-to-speech
speech_queue = queue.Queue()

# Function to process the speech queue
def speech_worker():
    while True:
        text = speech_queue.get()
        if text is None:  # Exit condition
            break
        engine.say(text)
        engine.runAndWait()
        speech_queue.task_done()

# Start the speech worker thread
speech_thread = threading.Thread(target=speech_worker, daemon=True)
speech_thread.start()

def speak(text):
    """Add text to the speech queue."""
    if not speech_queue.full():
        speech_queue.put(text)

# ORB Feature Detector and Matcher
orb = cv2.ORB_create(nfeatures=500)
bf = cv2.BFMatcher(cv2.NORM_HAMMING)

# Folder paths for templates
template_folders = {
    "5 Rupees": r"D:\PDL_FINAL\dataset\train\5_rupees",
    "10 Rupees": r"D:\PDL_FINAL\dataset\train\10_rupees",
    "20 Rupees": r"D:\PDL_FINAL\dataset\train\20_rupees",
    "50 Rupees": r"D:\PDL_FINAL\dataset\train\50_rupees",
    "100 Rupees": r"D:\PDL_FINAL\dataset\train\100_rupees",
    "200 Rupees": r"D:\PDL_FINAL\dataset\train\200_rupees",
    "500 Rupees": r"D:\PDL_FINAL\dataset\train\500_rupees",
}

# Load templates and compute ORB descriptors
def load_templates():
    """Load templates and compute ORB descriptors for each template image."""
    template_descriptors = {}
    for label, folder_path in template_folders.items():
        descriptors = []
        for image_file in os.listdir(folder_path):
            if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(folder_path, image_file)
                img = glib.readGrayImage(image_path)
                kp, desc = orb.detectAndCompute(img, None)
                if desc is not None:
                    descriptors.append(desc)
        template_descriptors[label] = descriptors
    return template_descriptors

template_descriptors = load_templates()

# Stability logic variables
last_detected = None
consecutive_frames = 0
frame_threshold = 3  # Number of consecutive frames needed for stable detection
cooldown_time = 3  # Cooldown time in seconds between announcements
last_announcement_time = 0  # Timestamp of the last announcement

# Function to process frame and detect currency
def process_frame(frame, result):
    global last_detected, consecutive_frames, last_announcement_time

    gray_frame = glib.imgToGray(frame)
    kp_frame, desc_frame = orb.detectAndCompute(gray_frame, None)

    if desc_frame is not None:
        best_match = None
        max_good_matches = 0
        good_match_threshold = 15  # Minimum number of good matches for a reliable detection

        for bill_name, descriptors_list in template_descriptors.items():
            for desc_template in descriptors_list:
                matches = bf.knnMatch(desc_template, desc_frame, k=2)
                good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]

                if len(good_matches) > max_good_matches:
                    max_good_matches = len(good_matches)
                    best_match = bill_name

        # Stability and confidence check
        if best_match and max_good_matches > good_match_threshold:
            if best_match == last_detected:
                consecutive_frames += 1
            else:
                last_detected = best_match
                consecutive_frames = 1

            if consecutive_frames >= frame_threshold:
                current_time = time.time()
                if current_time - last_announcement_time > cooldown_time:
                    result["text"] = f"Detected: {best_match}"
                    speak(f"{best_match} detected")
                    last_announcement_time = current_time  # Update last announcement time
                    consecutive_frames = 0  # Reset after a successful announcement
        else:
            last_detected = None
            consecutive_frames = 0
            result["text"] = ""  # Clear text if no valid match is found

# Start video capture

#if __name__ == "__main__":
cap = cv2.VideoCapture(0)
result = {"text": ""}
detection_thread = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if detection_thread is None or not detection_thread.is_alive():
        detection_thread = threading.Thread(target=process_frame, args=(frame.copy(), result))
        detection_thread.start()

    if result["text"]:
        cv2.putText(frame, result["text"], (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    glib.display_frame("Currency Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Stop the speech worker thread
speech_queue.put(None)
speech_thread.join()
