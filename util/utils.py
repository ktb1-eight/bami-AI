# utils.py

import pandas as pd
import numpy as np
import re
from typing import Union

def is_numeric(value: str) -> bool:
    """Check if the string value is numeric."""
    if pd.isna(value):
        return False
    return bool(re.match(r'^[0-9.]+$', value))

def calculate_mse_similarity(user1: Union[list, pd.Series, pd.DataFrame], user2: Union[list, pd.Series, pd.DataFrame]) -> float:
    """두 사용자 간의 평균 제곱 차이(MSE) 유사도를 계산하는 함수"""
    mse = np.mean((np.array(user1) - np.array(user2)) ** 2)
    return mse
