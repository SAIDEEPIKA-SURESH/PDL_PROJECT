from flask import Flask, render_template, Response
import cv2
import threading
import queue
import time
from lib import glib
from c7 import process_frame, load_templates, speak  # Ensure c7.py is in the same directory

app = Flask(__name__)

# Load currency templates
template_descriptors = load_templates()

# Queue for handling speech requests
speech_queue = queue.Queue()

# Function to handle speech in a separate thread
def speech_worker():
    while True:
        text = speech_queue.get()
        speak(text)
        speech_queue.task_done()

# Start speech processing thread
speech_thread = threading.Thread(target=speech_worker, daemon=True)
speech_thread.start()

# Stability Variables
last_detected = None
consecutive_frames = 0
frame_threshold = 3  # Number of stable detections before confirming
cooldown_time = 3  # Delay before repeating detection
last_announcement_time = 0

# Video streaming function
def generate_frames():
    global last_detected, consecutive_frames, last_announcement_time
    cap = cv2.VideoCapture(0)  # Use webcam

    while True:
        success, frame = cap.read()
        if not success:
            break

        # Process frame for currency detection
        result = {"text": ""}
        process_frame(frame, result)

        # If a currency is detected, announce it
        if result["text"]:
            current_time = time.time()
            if current_time - last_announcement_time > cooldown_time:
                speech_queue.put(result["text"])
                last_announcement_time = current_time  # Reset cooldown

            # Display the detected currency on the frame
            cv2.putText(frame, f"Detected: {result['text']}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Encode frame to send it to the frontend
        _, buffer = cv2.imencode(".jpg", frame)
        frame_bytes = buffer.tobytes()

        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    print("Starting Flask server...")  # Debugging log
    app.run(host="0.0.0.0", port=5000, debug=True)  # Allow external access