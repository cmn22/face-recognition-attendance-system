import numpy as np
import pandas as pd
import cv2
import redis
from insightface.app import FaceAnalysis
from sklearn.metrics import pairwise
from datetime import datetime
import os
from functools import lru_cache
from dotenv import load_dotenv

# ------------------------------
# Environment Configuration
# ------------------------------
# Load environment variables from .env file
load_dotenv()

# Retrieve Redis connection parameters from environment
hostname = os.getenv("REDIS_HOST")
portnumber = int(os.getenv("REDIS_PORT"))
password = os.getenv("REDIS_PASSWORD")

# ------------------------------
# Redis Connection Setup
# ------------------------------
# Create Redis connection pool for efficient connection reuse
REDIS_POOL = redis.ConnectionPool(
    host=hostname,
    port=portnumber,
    password=password,
    decode_responses=False  # Keep binary data as bytes
)

def get_redis_connection():
    """Return a Redis connection from the pool"""
    return redis.Redis(connection_pool=REDIS_POOL)

# ------------------------------
# Data Retrieval Functions
# ------------------------------
@lru_cache(maxsize=128)  # Cache up to 128 most recent queries
def retrive_data(name):
    """
    Retrieve face data from Redis and convert to DataFrame
    Returns empty DataFrame if no data found
    """
    r = get_redis_connection()
    retrive_dict = r.hgetall(name)
    
    if not retrive_dict:
        print(f"Warning: No data found in Redis for '{name}'. Initializing empty dataset.")
        return pd.DataFrame(columns=['Name', 'Role', 'facial_features'])

    # Convert Redis binary data to Pandas DataFrame
    retrive_series = pd.Series({
        k.decode(): np.frombuffer(v, dtype=np.float32) 
        for k, v in retrive_dict.items()
    })

    retrive_df = pd.DataFrame({
        'name_role': retrive_series.index,
        'facial_features': retrive_series.values
    })

    # Clean and split name@role strings
    retrive_df['name_role'] = retrive_df['name_role'].astype(str)
    retrive_df[['Name', 'Role']] = retrive_df['name_role'].str.split('@', expand=True)

    return retrive_df[['Name', 'Role', 'facial_features']]

# ------------------------------
# Face Recognition Setup
# ------------------------------
# Configure face analysis model with optimized settings
faceapp = FaceAnalysis(
    name='buffalo_sc', 
    root='insightface_model', 
    providers=['CPUExecutionProvider']  # Use CPU for inference
)
faceapp.prepare(
    ctx_id=0, 
    det_size=(320, 320),  # Smaller detection size for faster processing
    det_thresh=0.5        # Confidence threshold for face detection
)

# ------------------------------
# Matching Algorithm
# ------------------------------
def ml_search_algorithm(dataframe, feature_column, test_vector,
                       name_role=['Name', 'Role'], thresh=0.5):
    """
    Find best matching face using cosine similarity
    Returns ('Unknown', 'Unknown') if no match above threshold
    """
    X = np.vstack(dataframe[feature_column].values)
    similar = pairwise.cosine_similarity(X, test_vector.reshape(1, -1)).flatten()
    
    mask = similar >= thresh
    if mask.any():
        idx = similar[mask].argmax()
        return dataframe.loc[mask].iloc[idx][name_role]
    return 'Unknown', 'Unknown'

# ------------------------------
# Real-Time Prediction Class
# ------------------------------
class RealTimePred:
    """Handles real-time face recognition and attendance logging"""
    
    def __init__(self):
        self.reset_dict()
        self.r = get_redis_connection()
        self.last_log_time = {}  # Track last log time per person
        
    def reset_dict(self):
        """Reset attendance logs"""
        self.logs = {
            'name': [],
            'role': [],
            'current_time': []
        }
    
    def should_log_person(self, person_name, current_time):
        """
        Check if person should be logged again
        Prevents duplicate logs within 10 seconds
        """
        if person_name == 'Unknown':
            return False
            
        last_time = self.last_log_time.get(person_name)
        if last_time is None:
            self.last_log_time[person_name] = current_time
            return True
            
        time_diff = (current_time - last_time).total_seconds()
        if time_diff >= 10:
            self.last_log_time[person_name] = current_time
            return True
            
        return False
    
    def saveLogs_redis(self):
        """Save attendance logs to Redis"""
        if not self.logs['name']:
            self.reset_dict()
            return
        
        mask = np.array(self.logs['name']) != 'Unknown'
        if not mask.any():
            self.reset_dict()
            return
        
        encoded_data = [
            f"{name}@{role}@{ctime}"
            for name, role, ctime in zip(
                np.array(self.logs['name'])[mask],
                np.array(self.logs['role'])[mask],
                np.array(self.logs['current_time'])[mask]
            )
        ]
        
        if encoded_data:
            self.r.lpush('attendance:logs', *encoded_data)
        
        self.reset_dict()
    
    def face_prediction(self, test_image, dataframe, feature_column,
                       name_role=['Name', 'Role'], thresh=0.5):
        """
        Process image frame to:
        1. Detect faces
        2. Recognize individuals
        3. Log attendance
        4. Draw annotations
        """
        current_time = datetime.now()
        current_time_str = current_time.strftime("%Y-%m-%d %H:%M:%S")
        results = faceapp.get(test_image)
        
        for res in results:
            bbox = res['bbox'].astype(int)
            embeddings = res['embedding']
            
            person_name, person_role = ml_search_algorithm(
                dataframe, feature_column, embeddings, name_role, thresh
            )
            
            if self.should_log_person(person_name, current_time):
                self.logs['name'].append(person_name)
                self.logs['role'].append(person_role)
                self.logs['current_time'].append(current_time_str)
            
            color = (0, 255, 0) if person_name != 'Unknown' else (0, 0, 255)
            cv2.rectangle(test_image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 1)
            cv2.putText(test_image, person_name, (bbox[0], bbox[1]),
                       cv2.FONT_HERSHEY_DUPLEX, 0.7, color, 2)
            cv2.putText(test_image, current_time_str, (bbox[0], bbox[3] + 10),
                       cv2.FONT_HERSHEY_DUPLEX, 0.7, color, 2)
        
        return test_image

# ------------------------------
# Registration Form Class
# ------------------------------
class RegistrationForm:
    """Handles new user registration and face embedding storage"""
    
    def __init__(self):
        self.sample = 0
        self.r = get_redis_connection()
    
    def reset(self):
        """Reset sample counter"""
        self.sample = 0
    
    def get_embedding(self, frame):
        """
        Extract face embedding from frame
        Returns annotated frame and embedding vector
        """
        results = faceapp.get(frame, max_num=1)
        embeddings = None
        
        for res in results:
            self.sample += 1
            bbox = res['bbox'].astype(int)
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 1)
            cv2.putText(frame, f"samples = {self.sample}",
                       (bbox[0], bbox[1]), cv2.FONT_HERSHEY_DUPLEX,
                       0.6, (255, 255, 0), 2)
            embeddings = res['embedding']
        
        return frame, embeddings
    
    def save_data_in_redis_db(self, name, role):
        """
        Save face embedding to Redis
        Returns:
            True: Success
            'name_false': Invalid name
            'file_false': Missing embedding file
        """
        if not name or not name.strip():
            return 'name_false'
        
        if 'face_embedding.txt' not in os.listdir():
            return 'file_false'
        
        x_array = np.loadtxt('face_embedding.txt', dtype=np.float32)
        x_mean = x_array.reshape(-1, 512).mean(axis=0).astype(np.float32)
        
        self.r.hset('academy:register', f'{name.strip()}@{role}', x_mean.tobytes())
        
        os.remove('face_embedding.txt')
        self.reset()
        return True