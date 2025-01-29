# PYTHONPATH=$(pwd) pytest tests/

import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock
from face_rec import ml_search_algorithm

@pytest.fixture
def fake_dataframe():
    return pd.DataFrame({
        "Name": ["John Doe", "Jane Doe"],
        "Role": ["Student", "Teacher"],
        "facial_features": [np.random.rand(512), np.random.rand(512)]
    })

def test_ml_search_algorithm(fake_dataframe):
    test_vector = np.random.rand(512)
    name, role = ml_search_algorithm(fake_dataframe, "facial_features", test_vector, ["Name", "Role"], 0.5)

    assert isinstance(name, str)
    assert isinstance(role, str)
