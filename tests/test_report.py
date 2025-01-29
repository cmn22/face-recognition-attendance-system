# PYTHONPATH=$(pwd) pytest tests/

import pytest
import pandas as pd
from unittest.mock import MagicMock, patch
from face_rec import retrive_data

def test_retrieve_data():
    fake_redis = MagicMock()
    fake_redis.hgetall.return_value = {b'John Doe@Student': b'\x00' * 512}
    
    with patch("face_rec.get_redis_connection", return_value=fake_redis):
        df = retrive_data("academy:register")
        
    assert isinstance(df, pd.DataFrame)
    assert "Name" in df.columns
    assert "Role" in df.columns
