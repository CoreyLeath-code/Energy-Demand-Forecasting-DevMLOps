import pandas as pd
from src.data_preprocess import add_time_features

def test_time_features():
    df = pd.DataFrame({"timestamp": ["2024-01-01 00:00:00"]})
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = add_time_features(df)
    assert "hour" in df.columns
