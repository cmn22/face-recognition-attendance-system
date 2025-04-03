import streamlit as st
from Home import face_rec
from streamlit_webrtc import webrtc_streamer
import av
import time
from functools import lru_cache

# ==============================================
# UI Configuration
# ==============================================
st.subheader('Real-Time Attendance System')

# ==============================================
# Data Retrieval Setup
# ==============================================
@lru_cache(maxsize=1)
def get_redis_face_db():
    """Cache and retrieve face data from Redis database"""
    return face_rec.retrive_data(name='academy:register')

# Display loading state while fetching data
with st.spinner('Retrieving Data from Redis DB...'):    
    redis_face_db = get_redis_face_db()
    st.dataframe(redis_face_db)

st.success("Data successfully retrieved from Redis")

# ==============================================
# Real-Time Processing Configuration
# ==============================================
# Time interval for saving logs (in seconds)
LOG_SAVE_INTERVAL = 30  
last_save_time = time.time()
realtimepred = face_rec.RealTimePred()

def video_frame_callback(frame):
    """
    Processes each video frame to:
    1. Detect and recognize faces
    2. Track attendance
    3. Periodically save logs
    """
    global last_save_time
    
    # Convert video frame to numpy array
    img = frame.to_ndarray(format="bgr24")
    
    # Perform face recognition
    pred_img = realtimepred.face_prediction(
        img, 
        redis_face_db, 
        feature_column='facial_features', 
        name_role=['Name', 'Role'], 
        thresh=0.5
    )
    
    # Periodic saving of attendance logs
    current_time = time.time()
    if current_time - last_save_time >= LOG_SAVE_INTERVAL:
        realtimepred.saveLogs_redis()
        last_save_time = current_time
    
    return av.VideoFrame.from_ndarray(pred_img, format="bgr24")

# ==============================================
# WebRTC Video Stream Setup
# ==============================================
webrtc_streamer(
    key="realtimePrediction",
    video_frame_callback=video_frame_callback,
    rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    },
    media_stream_constraints={
        "video": True,
        "audio": False
    }
)