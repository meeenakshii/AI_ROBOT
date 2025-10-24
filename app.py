import os
import pickle
import time
import base64
import numpy as np
import face_recognition
import cv2
import requests
import speech_recognition as sr
from flask import Flask, render_template, redirect
from flask_socketio import SocketIO
from pydub import AudioSegment
from threading import Event
import io

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

# API Keys
groq_api_key = os.getenv("GROQ_API_KEY")
elevenlabs_api_key = os.getenv("ELEVENLABS_API_KEY")

GROQ_MODEL = "llama3-70b-8192"
ELEVENLABS_VOICE_ID = "21m00Tcm4TlvDq8ikWAM"  # Rachel voice - you can change this

system_prompt = (
    "As an ISRO and Planets Assistant, provide concise, engaging, one-line facts or insights about the Indian Space Research Organisation's (ISRO) latest and upcoming projects and the planets in our solar system. "
    "For ISRO projects, include: Chandrayaan-4 (a 2027 lunar sample-return mission to collect and analyze lunar soil), "
    "Gaganyaan (India's first crewed mission, with uncrewed tests in 2025 and a 2026 crewed flight for three astronauts), "
    "NISAR (NASA-ISRO Synthetic Aperture Radar, launching 2025 for Earth observation of climate and disasters), "
    "AstroSat-2 (a proposed multi-wavelength observatory to study stars and galaxies), "
    "Venus Orbiter Mission (planned for 2028 to explore Venus's atmosphere and surface), "
    "SPADEX (Space Docking Experiment, 2025, for satellite docking technology), "
    "XPoSat (X-ray Polarimeter Satellite, launched January 2024 for cosmic X-ray studies), "
    "Aditya-L1 (solar observatory at L1 point, operational since 2024 for solar studies), "
    "and Mangalyaan (Mars Orbiter Mission, launched 2013, studied Mars's surface and atmosphere). "
    "For planets, include: Mercury (smallest planet, with extreme temperatures from 427°C to -173°C), "
    "Venus (hottest planet at ~460°C, thick CO2 atmosphere, targeted by ISRO's 2028 mission), "
    "Earth (only known life-supporting planet, studied by NISAR for environmental changes), "
    "Mars (red planet with evidence of ancient water, explored by ISRO's Mangalyaan), "
    "Jupiter (largest planet, gas giant with 145 moons and a Great Red Spot storm), "
    "Saturn (gas giant with iconic rings and 83 named moons, observed by telescopes like AstroSat), "
    "Uranus (ice giant with faint rings and a tilted axis), "
    "and Neptune (windy ice giant with supersonic storms and 14 moons). "
    "Emphasize ISRO's cost-effectiveness, innovation, global collaborations, and contributions to space science, and connect planetary facts to ISRO's missions where relevant (e.g., Venus mission, Mangalyaan for Mars). "
    "Keep responses to 2-3 sentences maximum. "
    "Ensure responses are accurate, inspiring, and highlight the wonder of space exploration. "
    "Keep your tone warm, conversational, and enthusiastic - speak like a friendly space expert who loves sharing knowledge, not like a textbook."
)

# ==============================
# Face Encodings
# ==============================
ENCODINGS_FILE = os.path.join(os.path.dirname(__file__), "encodings.pkl")
if os.path.exists(ENCODINGS_FILE):
    with open(ENCODINGS_FILE, "rb") as f:
        known_face_encodings, known_face_names = pickle.load(f)
    print(f"[INFO] ✅ Loaded {len(known_face_names)} face encodings")
else:
    known_face_encodings, known_face_names = [], []
    print("[WARNING] ⚠️ No encodings.pkl found. Face recognition will not work.")

# ==============================
# Globals
# ==============================
listening = False
speak_done_event = Event()

recognizer = sr.Recognizer()
recognizer.energy_threshold = 500  # Increased for better noise filtering
recognizer.dynamic_energy_threshold = True
recognizer.pause_threshold = 1.2  # Wait for user to finish speaking
recognizer.phrase_threshold = 0.3
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
    print("[DEBUG] ✅ Client connected to SocketIO")

@socketio.on("speak_done")
def handle_speak_done():
    print("[DEBUG] ✅ Received speak_done from client")
    speak_done_event.set()

def recognize_face():
    """Recognize face using webcam with 10-second timeout."""
    video_capture = cv2.VideoCapture(0)
    user_name = None
    start_time = time.time()
    
    print("[DEBUG] 👤 Starting face recognition...")
    
    while time.time() - start_time < 10:
        ret, frame = video_capture.read()
        if not ret:
            continue
            
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb)
        encodings = face_recognition.face_encodings(rgb, face_locations)
        
        if len(face_locations) > 0:
            print(f"[DEBUG] 👤 Found {len(face_locations)} face(s)")
        
        for encoding in encodings:
            matches = face_recognition.compare_faces(known_face_encodings, encoding, tolerance=TOLERANCE)
            distances = face_recognition.face_distance(known_face_encodings, encoding)
            
            if True in matches:
                best = np.argmin(distances)
                user_name = known_face_names[best]
                print(f"[INFO] ✅ Recognized: {user_name}")
                break
                
        if user_name:
            break
            
    video_capture.release()
    return user_name

def transcribe_with_elevenlabs(audio_data):
    """Transcribe audio using ElevenLabs Speech-to-Text API."""
    try:
        # Convert AudioData to WAV bytes
        wav_bytes = audio_data.get_wav_data()
        
        # ElevenLabs STT endpoint
        url = "https://api.elevenlabs.io/v1/speech-to-text"
        headers = {
            "xi-api-key": elevenlabs_api_key
        }
        files = {
            "audio": ("audio.wav", wav_bytes, "audio/wav")
        }
        data = {
            "model_id": "eleven_multilingual_v2"
        }
        
        print("[DEBUG] 📤 Sending audio to ElevenLabs STT...")
        response = requests.post(url, headers=headers, files=files, data=data, timeout=30)
        response.raise_for_status()
        
        result = response.json()
        text = result.get("text", "").strip()
        
        if text:
            print(f"[DEBUG] ✅ ElevenLabs transcription successful")
            return text
        else:
            print("[DEBUG] ⚠️ ElevenLabs returned empty transcription")
            return None
            
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] ❌ ElevenLabs STT error: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"[ERROR] Response: {e.response.text}")
        return None
    except Exception as e:
        print(f"[ERROR] ❌ Transcription error: {e}")
        return None

def listen_to_user(retries=1):
    """Listen to user speech input with ElevenLabs transcription."""
    for attempt in range(retries + 1):
        try:
            with sr.Microphone() as source:
                print("[DEBUG] 🎤 Adjusting for ambient noise...")
                recognizer.adjust_for_ambient_noise(source, duration=0.5)
                
                socketio.emit("status", {"msg": "🎤 Listening... Speak now!"})
                print("=" * 60)
                print("🎤 LISTENING - Speak now...")
                print("=" * 60)
                
                # Listen for speech with longer timeout
                audio = recognizer.listen(source, timeout=10, phrase_time_limit=15)
                print("[DEBUG] ✅ Audio captured! Transcribing...")
                socketio.emit("status", {"msg": "🔄 Processing your speech..."})
                
                # Use ElevenLabs for transcription
                text = transcribe_with_elevenlabs(audio)
                
                if text and len(text.strip()) > 0:
                    print("=" * 60)
                    print(f"👤 YOU SAID: {text}")
                    print("=" * 60)
                    return text
                else:
                    print("[DEBUG] ⚠️ Empty transcription received")
                    if attempt < retries:
                        emit_speak("I heard something but couldn't make it out. Try speaking a bit louder?")
                    else:
                        return None
                        
        except sr.WaitTimeoutError:
            print("[DEBUG] ⏱️ No speech detected within timeout")
            if attempt < retries:
                emit_speak("I didn't hear anything. Are you still there? Try again!")
            else:
                return None
        except OSError as e:
            print(f"[ERROR] ❌ Microphone error: {e}")
            emit_speak("I'm having trouble with the microphone. Please check if it's connected properly.")
            return None
        except Exception as e:
            print(f"[ERROR] ❌ Listen error: {e}")
            if attempt < retries:
                emit_speak("Oops, something went wrong. Let's try that again!")
            else:
                return None
    
    return None

def ask_groq(user_input):
    """Get AI response from Groq API."""
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {groq_api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input}
        ],
        "temperature": 0.7,
        "max_tokens": 150
    }
    
    try:
        res = requests.post(url, headers=headers, json=data, timeout=30)
        res.raise_for_status()
        reply = res.json()["choices"][0]["message"]["content"].strip()
        print(f"[INFO] ✅ AI Response generated")
        return reply
        
    except Exception as e:
        print(f"[ERROR] ❌ Groq API failure: {e}")
        return "Oops, I'm having trouble connecting to my knowledge base right now. Let's try that again in a moment!"

def generate_elevenlabs_speech(text):
    """Generate speech using ElevenLabs TTS API."""
    try:
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_VOICE_ID}"
        headers = {
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
            "xi-api-key": elevenlabs_api_key
        }
        data = {
            "text": text,
            "model_id": "eleven_monolingual_v1",
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.75,
                "style": 0.5,
                "use_speaker_boost": True
            }
        }
        
        print("[DEBUG] 🔊 Generating ElevenLabs speech...")
        response = requests.post(url, json=data, headers=headers, timeout=30)
        response.raise_for_status()
        
        # Save audio
        with open("temp.mp3", "wb") as f:
            f.write(response.content)
        
        print("[DEBUG] ✅ ElevenLabs speech generated successfully")
        return True
        
    except Exception as e:
        print(f"[ERROR] ❌ ElevenLabs TTS error: {e}")
        return False

def emit_speak(msg):
    """Generate speech and emit to client with synchronized animations."""
    speak_done_event.clear()
    print(f"[DEBUG] 🗣️ Generating speech for: {msg[:50]}...")
    
    try:
        # Generate speech with ElevenLabs
        success = generate_elevenlabs_speech(msg)
        
        if not success:
            print("[ERROR] Failed to generate ElevenLabs speech")
            return
        
        # Load and process audio
        sound = AudioSegment.from_file("temp.mp3")
        
        # Convert to base64 for transmission
        with open("temp.mp3", "rb") as audio_file:
            audio_base64 = base64.b64encode(audio_file.read()).decode("utf-8")
        
        print("[DEBUG] 👄 Emitting mouth_move_start")
        socketio.emit("mouth_move_start")
        
        # Emit audio with lip sync data
        socketio.emit("speak", {
            "msg": msg,
            "audio": audio_base64,
            "animation": "Idle",
            "facialExpression": "smile",
            "lipsync": {
                "mouthCues": [
                    {"start": 0.0, "end": 0.2, "value": "A"},
                    {"start": 0.2, "end": 0.4, "value": "E"},
                    {"start": 0.4, "end": 0.6, "value": "O"}
                ]
            }
        })
        
        # Wait for audio to complete
        audio_duration = len(sound) / 1000
        print(f"[DEBUG] ⏱️ Audio duration: {audio_duration:.2f} seconds")
        time.sleep(audio_duration + 0.5)
        
        print("[DEBUG] 👄 Emitting mouth_move_stop")
        socketio.emit("mouth_move_stop")
        
        # Wait for client confirmation
        speak_done_event.wait(timeout=10)
        
    except Exception as e:
        print(f"[ERROR] ❌ Speech generation error: {e}")

@socketio.on("start-face")
def handle_start():
    """Handle start button - recognize face and begin conversation loop."""
    global listening
    print("\n" + "=" * 60)
    print("🚀 STARTING ASSISTANT SESSION")
    print("=" * 60)
    
    socketio.emit("status", {"msg": "👤 Recognizing face..."})
    
    # Recognize user
    name = recognize_face()
    
    if not name:
        print("[DEBUG] ❌ No face recognized")
        emit_speak("Hmm, I don't think we've met before! I couldn't recognize your face.")
        return
    
    # Personalized greeting
    greeting = (
        f"Hey {name}! Great to see you again! "
        f"I'm your ISRO and Planets Assistant. "
        f"Ask me anything about India's space missions or the fascinating planets in our solar system!"
    )
    
    print("=" * 60)
    print(f"🤖 ROBOT SAYS: {greeting}")
    print("=" * 60)
    emit_speak(greeting)
    
    # Conversation loop
    listening = True
    consecutive_failures = 0
    
    while listening:
        # Wait for robot to finish speaking before listening
        time.sleep(1)
        
        user_input = listen_to_user(retries=2)
        
        if not user_input:
            print("[DEBUG] ⚠️ No valid user input received")
            consecutive_failures += 1
            
            if consecutive_failures >= 3:
                print("=" * 60)
                print("⚠️ TOO MANY FAILED ATTEMPTS - ENDING SESSION")
                print("=" * 60)
                emit_speak("I'm having trouble hearing you. Let's try again when you're ready!")
                listening = False
                break
            
            continue
        
        # Reset failure counter on success
        consecutive_failures = 0
        
        socketio.emit("status", {"msg": f"💬 You: {user_input}"})
        
        # Check for exit keywords
        exit_words = ['goodbye', 'bye', 'stop', 'exit', 'quit', 'thank you', 'thanks']
        if any(word in user_input.lower() for word in exit_words):
            print("=" * 60)
            print("👋 USER REQUESTED EXIT - ENDING SESSION")
            print("=" * 60)
            emit_speak("It was wonderful talking with you! Keep looking up at the stars. See you next time!")
            listening = False
            break
        
        # Get AI response
        socketio.emit("status", {"msg": "🤔 Thinking..."})
        print("[DEBUG] 🤔 Getting AI response...")
        
        reply = ask_groq(user_input)
        
        print("=" * 60)
        print(f"🤖 ROBOT SAYS: {reply}")
        print("=" * 60)
        
        emit_speak(reply)

@socketio.on("stop_face")
def handle_stop():
    """Handle stop button - end conversation gracefully."""
    global listening
    print("[DEBUG] 🛑 stop_face event triggered")
    listening = False
    emit_speak("It was wonderful talking with you! Keep looking up at the stars. See you next time!")
    socketio.emit("status", {"msg": "⏹️ Stopped."})

# ==============================
# Run Server
# ==============================
if __name__ == "__main__":
    print("=" * 60)
    print("⚡ ISRO & Planets Assistant with ElevenLabs")
    print("=" * 60)
    print(f"🌐 Server: http://localhost:8001")
    print(f"📡 WebSocket: Ready")
    print(f"🎤 Speech-to-Text: ElevenLabs")
    print(f"🗣️  Text-to-Speech: ElevenLabs (Voice: Rachel)")
    print(f"👤 Face Recognition: {'Enabled' if known_face_names else 'Disabled'}")
    print(f"🔑 ElevenLabs API: {'Configured' if elevenlabs_api_key else 'NOT CONFIGURED!'}")
    print("=" * 60)
    
    if not elevenlabs_api_key:
        print("\n⚠️  WARNING: ELEVENLABS_API_KEY not found in .env file!")
        print("Please add: ELEVENLABS_API_KEY=your_api_key_here\n")
    
    socketio.run(app, host="0.0.0.0", port=8001)
