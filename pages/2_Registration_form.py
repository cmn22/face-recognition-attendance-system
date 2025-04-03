import streamlit as st
from Home import face_rec
import cv2
import numpy as np
from streamlit_webrtc import webrtc_streamer
import av

# --------------------------
# UI Configuration
# --------------------------
st.subheader('Registration Form')

# Initialize registration form handler
registration_form = face_rec.RegistrationForm()

# --------------------------
# User Information Collection
# --------------------------
# Input fields for user details
person_name = st.text_input(
    label='Name',
    placeholder='First & Last Name'
)

role = st.selectbox(
    label='Select your Role',
    options=('Student', 'Teacher')
)

# --------------------------
# Video Processing Function
# --------------------------
def video_callback_func(frame):
    """
    Callback function for processing video frames:
    1. Converts frame to numpy array
    2. Extracts facial embeddings
    3. Saves embeddings to temporary file
    """
    # Convert frame to BGR format numpy array
    img = frame.to_ndarray(format='bgr24')
    
    # Get facial embeddings and processed image
    reg_img, embedding = registration_form.get_embedding(img)
    
    # Save embeddings to temporary file if detected
    if embedding is not None:
        with open('face_embedding.txt', mode='ab') as f:
            np.savetxt(f, embedding)
    
    return av.VideoFrame.from_ndarray(reg_img, format='bgr24')

# --------------------------
# WebRTC Video Stream Configuration
# --------------------------
webrtc_streamer(
    key='registration',
    video_frame_callback=video_callback_func,
    rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    }
)

# --------------------------
# Registration Submission
# --------------------------
if st.button('Submit'):
    # Attempt to save data to Redis
    return_val = registration_form.save_data_in_redis_db(person_name, role)
    
    # Handle response cases
    if return_val is True:
        st.success(f"{person_name} registered successfully")
    elif return_val == 'name_false':
        st.error('Name cannot be empty or contain only spaces')
    elif return_val == 'file_false':
        st.error('No face embeddings found. Please refresh and try again.')