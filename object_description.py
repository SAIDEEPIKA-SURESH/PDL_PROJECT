import cv2
import google.generativeai as genai
from PIL import Image
import numpy as np
import pyttsx3
import io

# ✅ Set up Gemini API key
GENAI_API_KEY = "AIzaSyDpwUZes2zHdt1Bj8P8xutA2hM5gqk803M"  # 🔹 Replace with your actual API key
genai.configure(api_key=GENAI_API_KEY)

# ✅ Initialize text-to-speech engine
tts_engine = pyttsx3.init()
tts_engine.setProperty("rate", 150)  # Adjust speech speed

# ✅ Load the Gemini Vision model
model = genai.GenerativeModel(model_name="gemini-pro-vision")

def capture_frame():
    """Captures a frame from the webcam and returns it as an image."""
    cap = cv2.VideoCapture(0)  # Open webcam (0 for default camera)
    if not cap.isOpened():
        print("❌ Error: Could not open webcam.")
        return None
    
    ret, frame = cap.read()
    cap.release()  # Release camera after capturing one frame
    
    if not ret:
        print("❌ Error: Could not capture frame.")
        return None
    
    return frame

def describe_image(frame):
    """Processes the captured frame with Gemini API and returns descriptions."""
    try:
        # Convert OpenCV frame (NumPy array) to PIL image
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Convert image to byte array
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format="JPEG")
        img_byte_arr = img_byte_arr.getvalue()

        # 🔹 Send image to Gemini API for analysis
        response = model.generate_content(
            [image, "Describe the objects and environment in this image."]
        )

        # Extract description
        return response.text if response else "No description available."
    
    except Exception as e:
        return f"❌ Error: {str(e)}"

def speak(text):
    """Converts text to speech."""
    tts_engine.say(text)
    tts_engine.runAndWait()

def main():
    print("🎥 Starting real-time object and environment description...")
    
    while True:
        frame = capture_frame()  # Capture a frame
        
        if frame is not None:
            description = describe_image(frame)  # Get description from Gemini
            print("\n📝 Description:", description)  # Print description
            
            speak(description)  # Convert description to speech
            
        # Ask if the user wants to continue
        cont = input("🔄 Press Enter to capture again or type 'exit' to quit: ").strip().lower()
        if cont == "exit":
            break

if __name__ == "__main__":
    main()
