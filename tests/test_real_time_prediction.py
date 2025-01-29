# PYTHONPATH=$(pwd) pytest tests/

import pytest
import numpy as np
import cv2
from unittest.mock import MagicMock
from face_rec import RealTimePred

@pytest.fixture
def real_time_pred():
    return RealTimePred()

def test_should_log_person(real_time_pred):
    from datetime import datetime, timedelta
    
    current_time = datetime.now()
    assert real_time_pred.should_log_person("John Doe", current_time) == True

    real_time_pred.last_log_time["John Doe"] = current_time
    assert real_time_pred.should_log_person("John Doe", current_time + timedelta(seconds=5)) == False
    assert real_time_pred.should_log_person("John Doe", current_time + timedelta(seconds=15)) == True

def test_face_prediction(real_time_pred):
    fake_image = np.zeros((100, 100, 3), dtype=np.uint8)
    fake_df = MagicMock()
    
    processed_image = real_time_pred.face_prediction(fake_image, fake_df, 'facial_features', ['Name', 'Role'], 0.5)
    assert isinstance(processed_image, np.ndarray)
