# PYTHONPATH=$(pwd) pytest tests/

import pytest
import numpy as np
import os
from unittest.mock import patch, MagicMock
from face_rec import RegistrationForm

@pytest.fixture
def registration_form():
    return RegistrationForm()

def test_get_embedding(registration_form):
    fake_image = np.zeros((100, 100, 3), dtype=np.uint8)
    
    frame, embedding = registration_form.get_embedding(fake_image)
    assert isinstance(frame, np.ndarray)
    assert embedding is None or isinstance(embedding, np.ndarray)

def test_save_data_in_redis_db(registration_form):
    with patch("face_rec.redis.StrictRedis") as mock_redis:
        mock_redis_instance = mock_redis.return_value
        result = registration_form.save_data_in_redis_db("John Doe", "Student")
        
        assert result in [True, "name_false", "file_false"]
