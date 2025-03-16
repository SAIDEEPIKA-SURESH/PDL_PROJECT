from flask import Flask, render_template, Response
import os
import cv2
import numpy as np
import pyttsx3
import threading
import queue
import time
from lib import glib  # Assuming glib is your helper module

app = Flask(__name__)

# Initialize Video Capture
camera = cv2.VideoCapture(0)

# Initialize text-to-speech
engine = pyttsx3.init()
engine.setProperty('rate', 150)

# Queue for speech processing
speech_queue = queue.Queue()

def speech_worker():
    while True:
        text = speech_queue.get()
        if text is None:  
            break
        engine.say(text)
        engine.runAndWait()
        speech_queue.task_done()

speech_thread = threading.Thread(target=speech_worker, daemon=True)
speech_thread.start()

def speak(text):
    """Queue text for speech output."""
    if not speech_queue.full():
        speech_queue.put(text)

# ORB Feature Detector
orb = cv2.ORB_create(nfeatures=500)
bf = cv2.BFMatcher(cv2.NORM_HAMMING)

# Dataset folder paths
template_folders = {
    "5 Rupees": r"D:\PDL_FINAL\dataset\train\5_rupees",
    "10 Rupees": r"D:\PDL_FINAL\dataset\train\10_rupees",
    "20 Rupees": r"D:\PDL_FINAL\dataset\train\20_rupees",
    "50 Rupees": r"D:\PDL_FINAL\dataset\train\50_rupees",
    "100 Rupees": r"D:\PDL_FINAL\dataset\train\100_rupees",
    "200 Rupees": r"D:\PDL_FINAL\dataset\train\200_rupees",
    "500 Rupees": r"D:\PDL_FINAL\dataset\train\500_rupees",
}

# Load templates
def load_templates():
    """Load template images and extract ORB features."""
    template_descriptors = {}
    for label, folder_path in template_folders.items():
        if not os.path.exists(folder_path):
            print(f"⚠ Warning: Folder not found - {folder_path}")
            continue

        descriptors = []
        for image_file in os.listdir(folder_path):
            if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                img = glib.readGrayImage(os.path.join(folder_path, image_file))
                kp, desc = orb.detectAndCompute(img, None)
                if desc is not None:
                    descriptors.append(desc)
        template_descriptors[label] = descriptors
    return template_descriptors

template_descriptors = load_templates()

# Stability tracking
last_detected = None
consecutive_frames = 0
frame_threshold = 3
cooldown_time = 3  # Cooldown in seconds
last_announcement_time = 0

def process_frame(frame):
    """Detect currency and return annotated frame."""
    global last_detected, consecutive_frames, last_announcement_time

    gray_frame = glib.imgToGray(frame)
    kp_frame, desc_frame = orb.detectAndCompute(gray_frame, None)

    if desc_frame is not None:
        best_match = None
        max_good_matches = 0
        good_match_threshold = 15

        for bill_name, descriptors_list in template_descriptors.items():
            for desc_template in descriptors_list:
                matches = bf.knnMatch(desc_template, desc_frame, k=2)
                good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]

                if len(good_matches) > max_good_matches:
                    max_good_matches = len(good_matches)
                    best_match = bill_name

        # Stability check
        if best_match and max_good_matches > good_match_threshold:
            if best_match == last_detected:
                consecutive_frames += 1
            else:
                last_detected = best_match
                consecutive_frames = 1

            if consecutive_frames >= frame_threshold:
                current_time = time.time()
                if current_time - last_announcement_time > cooldown_time:
                    speak(f"{best_match} detected")
                    last_announcement_time = current_time
                    consecutive_frames = 0
                    cv2.putText(frame, f"Detected: {best_match}", (50, 50), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            last_detected = None
            consecutive_frames = 0

    return frame

# Video feed generator
def generate_frames():
    """Stream video frames with currency detection overlay."""
    while True:
        success, frame = camera.read()
        if not success:
            break
        frame = process_frame(frame)
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Flask Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# ...existing code...

if __name__ == "__main__":
    app.run(debug=True)