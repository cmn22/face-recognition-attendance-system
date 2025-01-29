import streamlit as st 
from Home import face_rec
from streamlit_webrtc import webrtc_streamer
import av
import time
from functools import lru_cache

st.subheader('Real-Time Attendance System')

# Cache the database retrieval
@lru_cache(maxsize=1)
def get_redis_face_db():
    return face_rec.retrive_data(name='academy:register')

# Retrieve the data from Redis Database
with st.spinner('Retrieving Data from Redis DB ...'):    
    redis_face_db = get_redis_face_db()
    st.dataframe(redis_face_db)

st.success("Data successfully retrieved from Redis")

# Initialize time tracking
WAIT_TIME = 30  # time in sec
last_save_time = time.time()
realtimepred = face_rec.RealTimePred()

def video_frame_callback(frame):
    global last_save_time
    
    # Convert frame efficiently
    img = frame.to_ndarray(format="bgr24")
    
    # Process frame
    pred_img = realtimepred.face_prediction(
        img, redis_face_db, 
        'facial_features', 
        ['Name', 'Role'], 
        thresh=0.5
    )
    
    # Check if it's time to save logs
    current_time = time.time()
    if current_time - last_save_time >= WAIT_TIME:
        realtimepred.saveLogs_redis()
        last_save_time = current_time
    
    return av.VideoFrame.from_ndarray(pred_img, format="bgr24")

# Initialize webRTC streamer
webrtc_streamer(
    key="realtimePrediction",
    video_frame_callback=video_frame_callback,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)