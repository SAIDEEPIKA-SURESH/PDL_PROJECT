import openai
import cv2
import numpy as np
import pyttsx3
import base64
from io import BytesIO

# Set your OpenAI API key
client = openai.OpenAI(api_key="your_openai_api_key")

# Initialize text-to-speech engine
engine = pyttsx3.init()

def encode_image(frame):
    """Encodes the frame to base64 for OpenAI API."""
    _, buffer = cv2.imencode(".jpg", frame)
    return base64.b64encode(buffer).decode("utf-8")

def describe_frame(frame):
    """Sends the captured frame to OpenAI's GPT-4 Vision for object detection and description."""
    image_base64 = encode_image(frame)

    response = client.chat.completions.create(
        model="gpt-4-vision-preview",
        messages=[
            {"role": "system", "content": "You are an AI that describes objects in an image and the surrounding environment."},
            {"role": "user", "content": [
                {"type": "text", "text": "Describe the objects in this image and provide an overview of the environment."},
                {"type": "image_url", "image_url": f"data:image/jpeg;base64,{image_base64}"}
            ]}
        ],
        max_tokens=500
    )
    
    description = response.choices[0].message.content
    return description

def speak(text):
    """Converts text to speech."""
    engine.say(text)
    engine.runAndWait()

# Start real-time detection
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    description = describe_frame(frame)
    print(description)
    speak(description)
    
    cv2.imshow("Live Object Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
