import os
import pickle
import time
import base64
import numpy as np
import face_recognition
import cv2
import requests
import speech_recognition as sr
from gtts import gTTS
from flask import Flask, render_template, redirect
from flask_socketio import SocketIO
from pydub import AudioSegment
from threading import Event

# ==============================
# Flask-SocketIO Setup
# ==============================
app = Flask(__name__, template_folder="templates", static_folder="static")
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

# ==============================
# Environment Setup
# ==============================
from dotenv import load_dotenv
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "llama3-70b-8192"

system_prompt = (
    "As an ISRO and Planets Assistant, provide concise, engaging, one-line facts..."
)

# ==============================
# Face Encodings
# ==============================
ENCODINGS_FILE = os.path.join(os.path.dirname(__file__), "encodings.pkl")
if os.path.exists(ENCODINGS_FILE):
    with open(ENCODINGS_FILE, "rb") as f:
        known_face_encodings, known_face_names = pickle.load(f)
else:
    known_face_encodings, known_face_names = [], []

# ==============================
# Globals
# ==============================
listening = False
speak_done_event = Event()

recognizer = sr.Recognizer()
recognizer.energy_threshold = 150
recognizer.dynamic_energy_threshold = True
TOLERANCE = 0.4

# ==============================
# ROUTES
# ==============================
@app.route("/")
def index():
    """Main page with button to open popup."""
    return render_template("index.html")

@app.route("/robot")
def robot_page():
    """Popup robot UI page."""
    return render_template("robot.html")

@app.errorhandler(404)
def page_not_found(e):
    """Redirect unknown URLs to main page to prevent Not Found."""
    return redirect("/")

# ==============================
# Socket.IO Events
# ==============================
@socketio.on("connect")
def handle_connect():
    print("[DEBUG] Client connected to SocketIO")

@socketio.on("speak_done")
def handle_speak_done():
    speak_done_event.set()

def recognize_face():
    video_capture = cv2.VideoCapture(0)
    user_name = None
    start_time = time.time()
    while time.time() - start_time < 10:
        ret, frame = video_capture.read()
        if not ret:
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb)
        encodings = face_recognition.face_encodings(rgb, face_locations)
        for encoding in encodings:
            matches = face_recognition.compare_faces(known_face_encodings, encoding, tolerance=TOLERANCE)
            distances = face_recognition.face_distance(known_face_encodings, encoding)
            if True in matches:
                best = np.argmin(distances)
                user_name = known_face_names[best]
                break
        if user_name:
            break
    video_capture.release()
    return user_name

def listen_to_user(retries=1):
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source, duration=1)
        socketio.emit("status", {"msg": "Listening..."})
        for attempt in range(retries + 1):
            try:
                audio = recognizer.listen(source, timeout=10, phrase_time_limit=15)
                text = recognizer.recognize_google(audio)
                return text
            except sr.UnknownValueError:
                if attempt < retries:
                    emit_speak("I didn't catch that. Please try again.")
                else:
                    return None
            except sr.WaitTimeoutError:
                return None

def ask_groq(user_input):
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    data = {"model": GROQ_MODEL, "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_input}]}
    try:
        res = requests.post(url, headers=headers, json=data)
        res.raise_for_status()
        return res.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print("[ERROR] Groq API failure:", e)
        return "Sorry, I couldn't get a response."

def emit_speak(msg):
    speak_done_event.clear()
    tts = gTTS(msg)
    tts.save("temp.mp3")
    sound = AudioSegment.from_file("temp.mp3")
    sound = sound.speedup(playback_speed=1.2)
    sound.export("temp.mp3", format="mp3")
    with open("temp.mp3", "rb") as audio_file:
        audio_base64 = base64.b64encode(audio_file.read()).decode("utf-8")
    socketio.emit("mouth_move_start")
    socketio.emit("speak", {"msg": msg, "audio": audio_base64, "animation": "Idle", "facialExpression": "smile", "lipsync": {"mouthCues": [{"start":0.0,"end":0.4,"value":"A"}]}})
    audio_duration = len(sound)/1000
    time.sleep(audio_duration + 0.5)
    socketio.emit("mouth_move_stop")
    speak_done_event.wait(timeout=10)

@socketio.on("start-face")
def handle_start():
    global listening
    socketio.emit("status", {"msg": "Recognizing face..."})
    name = recognize_face()
    if not name:
        emit_speak("Sorry, I could not recognize your face.")
        return
    emit_speak(f"Hi {name}! I'm your ISRO and Planets Assistant. Want to hear about ISRO’s missions or planets?")
    listening = True
    while listening:
        user_input = listen_to_user()
        if not user_input:
            emit_speak("Sorry, I didn't understand that.")
            continue
        reply = ask_groq(user_input)
        emit_speak(reply)

@socketio.on("stop_face")
def handle_stop():
    global listening
    listening = False
    emit_speak("Goodbye! See you later.")
    socketio.emit("status", {"msg": "Stopped."})

# ==============================
# Run Server
# ==============================
if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=8001)
