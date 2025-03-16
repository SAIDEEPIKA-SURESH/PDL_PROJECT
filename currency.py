from flask import Flask, render_template, Response
import cv2
import os
import numpy as np
import pyttsx3
import threading
import queue
import time

app = Flask(__name__)

# Initialize text-to-speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)

speech_queue = queue.Queue()
last_spoken = None  # To track last spoken currency
last_speak_time = 0  # Time tracking to avoid excessive repeats

# Speech worker thread
def speech_worker():
    while True:
        text = speech_queue.get()
        if text is None:
            break
        engine.say(text)
        engine.runAndWait()
        speech_queue.task_done()

# Start the speech worker thread
speech_thread = threading.Thread(target=speech_worker, daemon=True)
speech_thread.start()

def speak(text):
    """Queue text to be spoken"""
    global last_spoken, last_speak_time
    current_time = time.time()
    
    # Speak only if it's a new currency OR 3 seconds have passed since last speech
    if text != last_spoken or (current_time - last_speak_time > 3):
        last_spoken = text
        last_speak_time = current_time
        speech_queue.put(text)

# ORB Feature Detector
orb = cv2.ORB_create(nfeatures=500)
bf = cv2.BFMatcher(cv2.NORM_HAMMING)

# Folder paths for templates
template_folders = {
    "5 Rupees": r"D:\\PDL_FINAL\\dataset\\train\\5_rupees",
    "10 Rupees": r"D:\\PDL_FINAL\\dataset\\train\\10_rupees",
    "20 Rupees": r"D:\\PDL_FINAL\\dataset\\train\\20_rupees",
    "50 Rupees": r"D:\\PDL_FINAL\\dataset\\train\\50_rupees",
    "100 Rupees": r"D:\\PDL_FINAL\\dataset\\train\\100_rupees",
    "200 Rupees": r"D:\\PDL_FINAL\\dataset\\train\\200_rupees",
    "500 Rupees": r"D:\\PDL_FINAL\\dataset\\train\\500_rupees",
}

# Load template images and compute ORB descriptors
def load_templates():
    template_descriptors = {}
    for label, folder_path in template_folders.items():
        descriptors = []
        for image_file in os.listdir(folder_path):
            if image_file.lower().endswith((".png", ".jpg", ".jpeg")):
                image_path = os.path.join(folder_path, image_file)
                img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    kp, desc = orb.detectAndCompute(img, None)
                    if desc is not None:
                        descriptors.append(desc)
        template_descriptors[label] = descriptors
    return template_descriptors

template_descriptors = load_templates()

# Open webcam
cap = cv2.VideoCapture(0)

def generate_frames():
    global last_spoken
    while True:
        success, frame = cap.read()
        if not success:
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp_frame, desc_frame = orb.detectAndCompute(gray_frame, None)

        detected_currency = None

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

            if best_match and max_good_matches > good_match_threshold:
                detected_currency = best_match
                speak(f"{best_match} detected")  # Speak every time currency is found
            else:
                # Reset last_spoken if no currency is detected
                last_spoken = None

        # Draw detected currency on frame
        if detected_currency:
            cv2.putText(frame, f"Detected: {detected_currency}", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    engine.say("Currency detection system is now running.")
    engine.runAndWait()
    app.run(debug=True)