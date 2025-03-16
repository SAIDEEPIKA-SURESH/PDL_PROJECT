import os
import cv2
import numpy as np
from flask import Flask, render_template, Response
import pyttsx3
from detector import detect_currency

app = Flask(__name__)

# Initialize Text-to-Speech (TTS)
engine = pyttsx3.init()
engine.setProperty('rate', 140)

def speak(text):
    """Convert text to speech."""
    engine.say(text)
    engine.runAndWait()

# OpenCV Video Capture
camera = cv2.VideoCapture(0)

def generate_frames():
    """Capture frames from the camera and process them for currency detection."""
    while True:
        success, frame = camera.read()
        if not success:
            break

        # Convert frame to grayscale for processing
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Save frame temporarily
        cv2.imwrite("temp.jpg", gray)

        # Detect currency
        detected_currency = detect_currency("temp.jpg")

        # Speak detected currency
        speak(f"Detected {detected_currency}")

        # Display detected currency on frame
        cv2.putText(frame, detected_currency, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Encode frame for streaming
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
