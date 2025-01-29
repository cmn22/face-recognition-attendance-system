import numpy as np
import pandas as pd
import cv2
import redis
from insightface.app import FaceAnalysis
from sklearn.metrics import pairwise
from datetime import datetime
import os
from functools import lru_cache


# Connect to Redis Client
hostname = 'redis-12955.crce179.ap-south-1-1.ec2.redns.redis-cloud.com'
portnumber = 12955
password = '0AXzSpM3wa8utdJhuvvpRF8Iwc4TNVT3'

# Initialize the Redis connection pool first
REDIS_POOL = redis.ConnectionPool(
    host=hostname,
    port=portnumber,
    password=password,
    decode_responses=False
)

def get_redis_connection():
    return redis.Redis(connection_pool=REDIS_POOL)

# Optimize data retrieval with caching
@lru_cache(maxsize=128)
def retrive_data(name):
    r = get_redis_connection()
    retrive_dict = r.hgetall(name)
    
    # Vectorized operations instead of apply
    retrive_series = pd.Series({k.decode(): np.frombuffer(v, dtype=np.float32) 
                               for k, v in retrive_dict.items()})
    
    # More efficient DataFrame creation
    retrive_df = pd.DataFrame({
        'name_role': retrive_series.index,
        'facial_features': retrive_series.values
    })
    
    # Split name_role more efficiently
    retrive_df[['Name', 'Role']] = retrive_df['name_role'].str.split('@', expand=True)
    return retrive_df[['Name', 'Role', 'facial_features']]

# Configure face analysis with optimal settings
faceapp = FaceAnalysis(name='buffalo_sc', 
                      root='insightface_model', 
                      providers=['CPUExecutionProvider'])
faceapp.prepare(ctx_id=0, 
                det_size=(320, 320),  # Reduced from 640x640 for speed
                det_thresh=0.5)

# Optimized ML search algorithm
def ml_search_algorithm(dataframe, feature_column, test_vector,
                       name_role=['Name', 'Role'], thresh=0.5):
    """Optimized cosine similarity search algorithm"""
    # Convert list to array once
    X = np.vstack(dataframe[feature_column].values)
    
    # Compute cosine similarity efficiently
    similar = pairwise.cosine_similarity(X, test_vector.reshape(1, -1)).flatten()
    
    # Use numpy operations instead of pandas where possible
    mask = similar >= thresh
    if mask.any():
        idx = similar[mask].argmax()
        return dataframe.loc[mask].iloc[idx][name_role]
    return 'Unknown', 'Unknown'

class RealTimePred:
    def __init__(self):
        self.reset_dict()
        self.r = get_redis_connection()
        self.last_log_time = {}  # Track last log time for each person
        
    def reset_dict(self):
        self.logs = {
            'name': [],
            'role': [],
            'current_time': []
        }
    
    def should_log_person(self, person_name, current_time):
        """Check if enough time has passed to log this person again"""
        if person_name == 'Unknown':
            return False
            
        last_time = self.last_log_time.get(person_name)
        if last_time is None:
            self.last_log_time[person_name] = current_time
            return True
            
        time_diff = (current_time - last_time).total_seconds()
        if time_diff >= 10:  # Only log if 10 seconds have passed
            self.last_log_time[person_name] = current_time
            return True
            
        return False
    
    def saveLogs_redis(self):
        if not self.logs['name']:
            self.reset_dict()
            return
        
        # Process only non-Unknown entries
        mask = np.array(self.logs['name']) != 'Unknown'
        if not mask.any():
            self.reset_dict()
            return
        
        # Create encoded data efficiently
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
        current_time = datetime.now()
        current_time_str = current_time.strftime("%Y-%m-%d %H:%M:%S")
        results = faceapp.get(test_image)
        
        # Process all faces in parallel where possible
        for res in results:
            bbox = res['bbox'].astype(int)
            embeddings = res['embedding']
            
            person_name, person_role = ml_search_algorithm(
                dataframe, feature_column, embeddings, name_role, thresh
            )
            
            # Only log if enough time has passed for this person
            if self.should_log_person(person_name, current_time):
                # Append to logs
                self.logs['name'].append(person_name)
                self.logs['role'].append(person_role)
                self.logs['current_time'].append(current_time_str)
            
            # Draw rectangles and text
            color = (0, 255, 0) if person_name != 'Unknown' else (0, 0, 255)
            cv2.rectangle(test_image, (bbox[0], bbox[1]), 
                         (bbox[2], bbox[3]), color, 1)
            cv2.putText(test_image, person_name, (bbox[0], bbox[1]),
                       cv2.FONT_HERSHEY_DUPLEX, 0.7, color, 2)
            cv2.putText(test_image, current_time_str, (bbox[0], bbox[3] + 10),
                       cv2.FONT_HERSHEY_DUPLEX, 0.7, color, 2)
        
        return test_image

class RegistrationForm:
    def __init__(self):
        self.sample = 0
        self.r = get_redis_connection()
    
    def reset(self):
        self.sample = 0
    
    def get_embedding(self, frame):
        results = faceapp.get(frame, max_num=1)
        embeddings = None
        
        for res in results:
            self.sample += 1
            bbox = res['bbox'].astype(int)
            cv2.rectangle(frame, (bbox[0], bbox[1]), 
                         (bbox[2], bbox[3]), (0, 255, 0), 1)
            cv2.putText(frame, f"samples = {self.sample}",
                       (bbox[0], bbox[1]), cv2.FONT_HERSHEY_DUPLEX,
                       0.6, (255, 255, 0), 2)
            embeddings = res['embedding']
        
        return frame, embeddings
    
    def save_data_in_redis_db(self, name, role):
        if not name or not name.strip():
            return 'name_false'
        
        if 'face_embedding.txt' not in os.listdir():
            return 'file_false'
        
        # Load and process embeddings efficiently
        x_array = np.loadtxt('face_embedding.txt', dtype=np.float32)
        x_mean = x_array.reshape(-1, 512).mean(axis=0).astype(np.float32)
        
        # Save to Redis
        self.r.hset('academy:register',
                    f'{name.strip()}@{role}',
                    x_mean.tobytes())
        
        os.remove('face_embedding.txt')
        self.reset()
        return True