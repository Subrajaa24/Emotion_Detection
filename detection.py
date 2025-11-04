# detection_dashboard_report.py - FULL CODE with DB, Alerts, and Stability Fix
import streamlit as st
import cv2
from deepface import DeepFace
import pandas as pd
from collections import Counter
import numpy as np
from datetime import datetime, timedelta
import io
import sqlite3
import time # Added for controlled loop delay

st.set_page_config(page_title="🎭 Emotion Detection Dashboard", layout="wide")

# --- Database Setup ---
DB_FILE = "emotion_reports.db"

def init_db():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            start_time TEXT NOT NULL,
            emotion_data TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()

def save_report_to_db(timestamps, emotion_history):
    """Saves all collected emotion data as a single report record."""
    if not emotion_history:
        return None

    df_report = pd.DataFrame({
        "Timestamp": timestamps,
        "Emotion": emotion_history
    })
    
    # Store the report as a JSON string
    emotion_data_json = df_report.to_json(orient='records')
    start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO reports (start_time, emotion_data) VALUES (?, ?)",
        (start_time, emotion_data_json)
    )
    conn.commit()
    conn.close()
    return start_time

def get_past_reports():
    conn = sqlite3.connect(DB_FILE)
    df = pd.read_sql_query("SELECT id, start_time FROM reports ORDER BY id DESC", conn)
    conn.close()
    return df

# Initialize the database when the script starts
init_db()

# --- Configuration Constants ---
NEGATIVE_EMOTIONS = ['sad', 'angry', 'fear', 'disgust']
POSITIVE_EMOTION = 'happy'
ALERT_DURATION_SECONDS = 10 
FOCUS_EMOTIONS = ['neutral', 'contempt'] 
MOTION_THRESHOLD = 0.5 
COGNITIVE_DURATION_SECONDS = 10 
# Temporal Smoothing Constant
STABILITY_THRESHOLD = 10  
# -----------------------------

# --- Initialize Session State for ALL Variables ---
if 'emotion_history' not in st.session_state:
    st.session_state.emotion_history = []
if 'timestamps' not in st.session_state:
    st.session_state.timestamps = []
if 'is_running' not in st.session_state:
    st.session_state.is_running = False
if 'cap' not in st.session_state:
    st.session_state.cap = None

# Sustained Monitoring and Cognitive Tracking
if 'negative_state' not in st.session_state:
    st.session_state.negative_state = {'current_emotion': None, 'start_time': None, 'alert_triggered': False}
if 'positive_state' not in st.session_state:
    st.session_state.positive_state = {'current_emotion': None, 'start_time': None, 'message_displayed': False}
if 'cognitive_state' not in st.session_state:
    st.session_state.cognitive_state = {'current_state': 'N/A', 'start_time': None, 'alert_triggered': False}

# Variables for Temporal Smoothing (Flicker Fix)
if 'stable_emotion' not in st.session_state:
    st.session_state.stable_emotion = "Detecting..."
if 'stable_count' not in st.session_state:
    st.session_state.stable_count = 0
if 'previous_gray_frame' not in st.session_state:
    st.session_state.previous_gray_frame = None

# ---------------- Sidebar ----------------
st.sidebar.title("Settings")
show_fps = st.sidebar.checkbox("Show FPS", value=True)
resize_factor = st.sidebar.slider("Frame Resize Factor (Speed vs Accuracy)", 0.3, 1.0, 0.5)

st.sidebar.markdown("---")
st.sidebar.header("Ethics and Consent 🤝")

# --- ETHICAL DATA HANDLING: CONSENT ---
consent_given = st.sidebar.checkbox(
    "I consent to the analysis and local storage of my non-identifiable emotional data.",
    value=False
)

st.sidebar.markdown(
    """
    **Data Policy:**
    - **No video frames or identifiable images are saved.**
    - **Only** statistical data (time, emotion, motion score) is stored.
    - Data is stored **locally** in the `emotion_reports.db` file.
    - To delete all history, **delete the `emotion_reports.db` file** from the project folder.
    """
)
st.sidebar.markdown("---")

# Use key to allow Streamlit to track button state correctly
start_camera = st.sidebar.button("Start Camera", key="start_btn")
stop_camera = st.sidebar.button("Stop Camera", key="stop_btn")

# --- Sustained Monitoring System Info ---
st.sidebar.header("Sustained Monitoring System ⏱️")
st.sidebar.info(f"🚨 **Alert** if: **{', '.join([e.upper() for e in NEGATIVE_EMOTIONS])}** for {ALERT_DURATION_SECONDS} secs.")
st.sidebar.success(f"😊 **Reinforcement** if: **{POSITIVE_EMOTION.upper()}** for {ALERT_DURATION_SECONDS} secs.")
st.sidebar.warning(f"🧠 **Focus** inferred from **{', '.join([e.upper() for e in FOCUS_EMOTIONS])}** and low motion (<{MOTION_THRESHOLD}%) for {COGNITIVE_DURATION_SECONDS} secs.")
# --- End Alert System Controls ---

# ---------------- Placeholders ----------------
video_placeholder = st.empty()
chart_placeholder = st.empty()
dominant_emotion_placeholder = st.empty()
sustained_message_placeholder = st.empty()
cognitive_state_placeholder = st.empty() 

# ---------------- Helper Function ----------------
def process_frame(frame, resize_factor):
    small_frame = cv2.resize(frame, (0,0), fx=resize_factor, fy=resize_factor)
    motion_score = 0.0 # Default/initial motion score
    emotion = "Detecting..."
    
    try:
        # DeepFace analysis
        result = DeepFace.analyze(small_frame, actions=['emotion'], enforce_detection=False, silent=True)
        emotion = result[0]['dominant_emotion']
        
        # Motion Detection
        current_gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if st.session_state.previous_gray_frame is not None:
            # Calculate absolute difference between the current and previous frame
            frame_diff = cv2.absdiff(current_gray_frame, st.session_state.previous_gray_frame)
            # Threshold the difference (pixels brighter than 25 are considered motion)
            _, thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)
            # Calculate percentage of pixels that moved (motion score)
            motion_score = (np.sum(thresh) / 255.0) / (thresh.shape[0] * thresh.shape[1]) * 100
        
        st.session_state.previous_gray_frame = current_gray_frame
        
        # Annotation on frame
        cv2.putText(frame, f"Emotion: {emotion}", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Motion: {motion_score:.2f}%", (50, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        
    except:
        emotion = "Detecting..."
        cv2.putText(frame, emotion, (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        st.session_state.previous_gray_frame = None 
        motion_score = 0.0
        
    return frame, emotion, motion_score

# ---------------- START CAMERA LOGIC ----------------
if start_camera and not st.session_state.is_running:
    if not consent_given:
        st.error("⚠️ **Ethical Requirement:** You must grant consent to start the camera and begin analysis.")
    else:
        try:
            # Initialize camera in session state
            st.session_state.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        except:
            st.session_state.cap = cv2.VideoCapture(0) # Fallback

        if not st.session_state.cap.isOpened():
            st.error("❌ Cannot access the webcam. Please check permissions and device availability.")
            if st.session_state.cap:
                st.session_state.cap.release()
                st.session_state.cap = None
        else:
            # Reset states and start flag
            st.session_state.is_running = True 
            st.session_state.emotion_history = []
            st.session_state.timestamps = []
            st.session_state.stable_emotion = "Detecting..."
            st.session_state.stable_count = 0
            st.session_state.previous_gray_frame = None
            
            st.rerun()

# ---------------- STOP CAMERA and SAVE LOGIC ----------------
# Logic runs if the button is clicked OR if the stream fails inside the main loop
if stop_camera or (st.session_state.cap and not st.session_state.is_running and st.session_state.cap.isOpened()):
    
    if st.session_state.cap is not None:
        
        # Ensure running flag is set to False
        st.session_state.is_running = False
        
        # Release the camera resource
        if st.session_state.cap and st.session_state.cap.isOpened():
            st.session_state.cap.release()
            
        # SAVE THE ENTIRE SESSION HISTORY
        emotion_history = st.session_state.emotion_history
        timestamps = st.session_state.timestamps
        
        if emotion_history:
            saved_time = save_report_to_db(timestamps, emotion_history)
            if saved_time:
                st.success(f"✅ **FULL SESSION REPORT SAVED:** Session started at {saved_time} has been stored as a single record.")
                
            # Clear the session state history after saving
            st.session_state.emotion_history = []
            st.session_state.timestamps = []

        st.session_state.cap = None # Ensure cap is fully reset
        st.success("🛑 Camera stopped.")
        # Rerun once to clean up the interface
        st.rerun()

# ---------------- MAIN VIDEO PROCESSING LOOP ----------------
if st.session_state.get('is_running', False) and st.session_state.cap is not None:
    
    video_placeholder_live = video_placeholder.empty()
    cap = st.session_state.cap

    # Run the loop until st.session_state.is_running is set to False
    while st.session_state.get('is_running', False) and cap.isOpened():
        
        ret, frame = cap.read()
        current_time = datetime.now()
        
        if not ret:
            st.warning("⚠️ Cannot grab frame. Stopping stream.")
            st.session_state.is_running = False 
            break
            
        # Process the frame
        frame = cv2.flip(frame, 1)
        annotated_frame, current_emotion, motion_score = process_frame(frame, resize_factor)

        # Record emotions and timestamps into Session State
        st.session_state.emotion_history.append(current_emotion)
        st.session_state.timestamps.append(current_time.strftime("%H:%M:%S"))

        # --- Temporal Smoothing Logic (Flicker Fix) ---
        if current_emotion == st.session_state.stable_emotion:
            st.session_state.stable_count += 1
        else:
            st.session_state.stable_emotion = current_emotion
            st.session_state.stable_count = 1
        
        # Determine the emotion to display
        display_emotion = st.session_state.stable_emotion 
        if st.session_state.stable_count < STABILITY_THRESHOLD:
             # If not stable, show the previously stable emotion to prevent flicker
            pass # Keep previous display_emotion (which is st.session_state.stable_emotion)

        # --- Alert/Cognitive Logic ---
        negative_state = st.session_state.negative_state
        positive_state = st.session_state.positive_state
        cognitive_state = st.session_state.cognitive_state

        # Cognitive State Inference Logic (Focus/Fatigue)
        if current_emotion in FOCUS_EMOTIONS and motion_score <= MOTION_THRESHOLD:
            if cognitive_state['current_state'] != 'Focus':
                cognitive_state['current_state'] = 'Focus'
                cognitive_state['start_time'] = current_time
                cognitive_state['alert_triggered'] = False
            
            if not cognitive_state['alert_triggered'] and cognitive_state['start_time']:
                duration = current_time - cognitive_state['start_time']
                if duration >= timedelta(seconds=COGNITIVE_DURATION_SECONDS):
                    cognitive_state_placeholder.success("🎯 **Deep Focus:** Excellent stability and engagement!")
                    cognitive_state['alert_triggered'] = True
                else:
                    remaining = COGNITIVE_DURATION_SECONDS - duration.total_seconds()
                    cognitive_state_placeholder.info(f"Focusing... {remaining:.0f} seconds to confirmation.")
        
        elif current_emotion in ['sad', 'disgust', 'fear', 'angry'] and motion_score >= 2.0: 
            if cognitive_state['current_state'] != 'Fatigue/Distraction':
                cognitive_state['current_state'] = 'Fatigue/Distraction'
                cognitive_state['start_time'] = current_time
                cognitive_state['alert_triggered'] = False
            
            if not cognitive_state['alert_triggered'] and cognitive_state['start_time']:
                duration = current_time - cognitive_state['start_time']
                if duration >= timedelta(seconds=COGNITIVE_DURATION_SECONDS):
                    cognitive_state_placeholder.error("😴 **Fatigue Warning:** Disengagement or stress detected. Consider a break.")
                    cognitive_state['alert_triggered'] = True
        else:
            cognitive_state_placeholder.empty()
            st.session_state.cognitive_state = {'current_state': 'N/A', 'start_time': None, 'alert_triggered': False}

        # Sustained Emotion Alert/Message Logic (Negative)
        if current_emotion in NEGATIVE_EMOTIONS:
            if negative_state['current_emotion'] != current_emotion:
                negative_state['current_emotion'] = current_emotion
                negative_state['start_time'] = current_time
                negative_state['alert_triggered'] = False
            
            if not negative_state['alert_triggered'] and negative_state['start_time']:
                duration = current_time - negative_state['start_time']
                if duration >= timedelta(seconds=ALERT_DURATION_SECONDS):
                    alert_message = f"🚨 **ALERT!** The member has been **{current_emotion.upper()}** for past **{ALERT_DURATION_SECONDS} seconds**!"
                    sustained_message_placeholder.error(alert_message)
                    negative_state['alert_triggered'] = True
                else:
                    remaining = ALERT_DURATION_SECONDS - duration.total_seconds()
                    if not positive_state['message_displayed']:
                        sustained_message_placeholder.warning(f"Monitoring **{current_emotion.upper()}**... {remaining:.0f} seconds to alert.")
            
            # Reset positive state
            st.session_state.positive_state = {'current_emotion': None, 'start_time': None, 'message_displayed': False}

        # Sustained Emotion Alert/Message Logic (Positive)
        elif current_emotion == POSITIVE_EMOTION:
            if positive_state['current_emotion'] != POSITIVE_EMOTION:
                positive_state['current_emotion'] = POSITIVE_EMOTION
                positive_state['start_time'] = current_time
                positive_state['message_displayed'] = False
            
            if not positive_state['message_displayed'] and positive_state['start_time']:
                duration = current_time - positive_state['start_time']
                if duration >= timedelta(seconds=ALERT_DURATION_SECONDS):
                    message = f"🌟 **AWESOME!** You've been **HAPPY** for **{ALERT_DURATION_SECONDS} seconds**! **Keep Going!**"
                    sustained_message_placeholder.success(message)
                    positive_state['message_displayed'] = True
                else:
                    remaining = ALERT_DURATION_SECONDS - duration.total_seconds()
                    if not negative_state['alert_triggered']:
                        sustained_message_placeholder.info(f"Monitoring **{POSITIVE_EMOTION.upper()}**... {remaining:.0f} seconds to encouragement.")
            
            # Reset negative state
            st.session_state.negative_state = {'current_emotion': None, 'start_time': None, 'alert_triggered': False}

        # Reset states for neutral/surprise/contempt (emotions that stop the timer)
        elif current_emotion != "Detecting..." and current_emotion != POSITIVE_EMOTION and current_emotion not in NEGATIVE_EMOTIONS:
            if negative_state['current_emotion'] or positive_state['current_emotion']:
                sustained_message_placeholder.empty()
            st.session_state.negative_state = {'current_emotion': None, 'start_time': None, 'alert_triggered': False}
            st.session_state.positive_state = {'current_emotion': None, 'start_time': None, 'message_displayed': False}
            
        elif current_emotion == "Detecting...":
             # Clear messages if face is lost
            sustained_message_placeholder.empty()
            st.session_state.negative_state = {'current_emotion': None, 'start_time': None, 'alert_triggered': False}
            st.session_state.positive_state = {'current_emotion': None, 'start_time': None, 'message_displayed': False}


        # FPS and Display
        fps_value = int(cap.get(cv2.CAP_PROP_FPS)) if cap.get(cv2.CAP_PROP_FPS) > 0 else 30 
        if show_fps:
            cv2.putText(annotated_frame, f"FPS: {fps_value}", (50, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        # 1. Update the Video Frame
        frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        video_placeholder_live.image(frame_rgb, channels="RGB", use_container_width=True)
        
        # 2. Update Dominant Emotion (Uses the stabilized emotion)
        dominant_emotion_placeholder.markdown(f"### Dominant Emotion: **{display_emotion.upper()}**")
        
        # 3. Update Chart
        counts = Counter([e for e in st.session_state.emotion_history if e != "Detecting..."])
        if counts:
            df_chart = pd.DataFrame(list(counts.items()), columns=["Emotion", "Count"]).sort_values("Count", ascending=False)
            chart_placeholder.bar_chart(df_chart.set_index("Emotion"))
        
        # Introduce a controlled delay to limit the refresh rate (e.g., max 10 FPS = 0.1s)
        time.sleep(0.1) 
    
    # After the while loop breaks, force a Streamlit rerun
    st.rerun() 
    
# ---------------- Past Reports and Download ----------------
st.sidebar.markdown("---")
st.sidebar.header("📁 Past Reports History")

past_reports_df = get_past_reports()

if not past_reports_df.empty:
    report_options = {row['id']: row['start_time'] for index, row in past_reports_df.iterrows()}
    
    selected_report_id = st.sidebar.selectbox(
        "Select a report to download:",
        options=list(report_options.keys()),
        format_func=lambda x: report_options[x]
    )

    if selected_report_id:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute("SELECT emotion_data, start_time FROM reports WHERE id = ?", (selected_report_id,))
        result = cursor.fetchone()
        conn.close()

        if result:
            emotion_data_json, start_time = result
            df_past_report = pd.read_json(emotion_data_json, orient='records')

            csv_buffer = io.StringIO()
            df_past_report.to_csv(csv_buffer, index=False)
            csv_bytes = csv_buffer.getvalue().encode()
            
            st.sidebar.download_button(
                label=f"📥 Download Report: {start_time}",
                data=csv_bytes,
                file_name=f"emotion_report_{start_time.replace(' ', '_').replace(':', '')}.csv",
                mime="text/csv"
            )
else:
    st.sidebar.info("No past reports found.")