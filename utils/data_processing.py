# data_processing.py
import sys
import os
sys.path.append(os.path.pardir)

import pandas as pd
import numpy as np
from typing import List, Tuple
from util.utils import calculate_mse_similarity

def get_season(date: pd.Timestamp) -> str:
    """Return Season"""
    if date.month in (3, 4, 5):
        return '봄'
    elif date.month in (6, 7, 8):
        return '여름'
    elif date.month in (9, 10, 11):
        return '가을'
    else:
        return '겨울'

def get_user_travel_style(style_mapping: List[str]) -> List[int]:
    """사용자의 여행 스타일 데이터를 입력받는 함수"""
    user_travel_style = []
    
    for question in style_mapping:
        while True:
            try:
                preference = int(input(f"{question}에 대한 선호도를 입력하세요 (1부터 7까지의 값): "))
                if 1 <= preference <= 7:
                    user_travel_style.append(preference)
                    break
                else:
                    print("값은 1부터 7까지의 숫자여야 합니다.")
            except ValueError:
                print("유효한 숫자를 입력하세요.")
                
    return user_travel_style

def calculate_similarity(users: pd.DataFrame, user_travel_style: List[int], vehicle_usage: str) -> List[Tuple[str, float]]:
    """여행객 데이터와 사용자의 여행 스타일을 기반으로 유사도를 계산하는 함수"""
    vehicle_type = '자가용' if vehicle_usage == 'Y' else '대중교통 등'
    selected_users = users[users['MVMN_NM'] == vehicle_type]
    
    similarities = [
        (row['TRAVELER_ID'], calculate_mse_similarity(user_travel_style, row[[f'TRAVEL_STYL_{n}' for n in range(1, 8)]]))
        for _, row in selected_users.iterrows()
    ]
    
    return similarities

def find_nearest_attractions(atr_data: np.ndarray) -> np.ndarray:
    """
    시작점을 숙소로 설정 후 가장 가까운 어트랙션 찾는 과정을 반복하는 함수.
    
    Parameters:
    - atr_data: np.ndarray, 어트랙션 데이터 배열 (어트랙션명, 위도, 경도)
    
    Returns:
    - np.ndarray, 순서대로 정렬된 어트랙션 데이터 배열
    """
    atr_series = []
    current_location = atr_data[0, 1:].astype(np.float32)  # 시작점 좌표
    remaining_data = atr_data[1:]  # 시작점을 제외한 나머지 데이터

    while remaining_data.shape[0] > 0:
        # 나머지 모든 점과의 거리 계산
        locations = remaining_data[:, 1:].astype(np.float32)
        distances = np.sqrt(np.sum((locations - current_location) ** 2, axis=1))

        # 가장 가까운 점 찾기
        nearest_index = np.argmin(distances)
        nearest_attraction = remaining_data[nearest_index]

        # 현재 위치를 가장 가까운 점으로 업데이트
        current_location = nearest_attraction[1:].astype(np.float32)
        atr_series.append(nearest_attraction)

        # 방문한 점을 remaining_data에서 제거
        remaining_data = np.delete(remaining_data, nearest_index, axis=0)

    return np.array(atr_series)
