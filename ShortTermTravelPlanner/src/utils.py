from typing import Dict, Any
import joblib
import pandas as pd
import numpy as np


def process_user_input(user_input: Dict[str, Any]) -> pd.DataFrame:
    """
    사용자 입력 딕셔너리를 받아 pandas DataFrame으로 변환하고 필요한 전처리를 수행
    Args:
        user_input (Dict[str, Any]): 사용자 입력 데이터
    Returns:
        pd.DataFrame: 전처리된 사용자 입력 데이터프레임
    """

    user_df = pd.DataFrame([user_input])

    # 범주형 변수들 처리
    categorical_features = [
        'MVMN_NM', 'GENDER', 'REL_CD_Categorized', 'TRAVEL_MOTIVE_1'
    ]

    user_df[categorical_features] = user_df[categorical_features].astype(str)
    return user_df


def filter_and_merge(user_df: pd.DataFrame, info: pd.DataFrame) -> pd.DataFrame:
    """
    사용자 정보를 바탕으로 여행지 정보를 필터링하고 병합
    Args:
        user_df (pd.DataFrame): 사용자 입력 데이터
        info (pd.DataFrame): 여행지 정보
    Returns:
        pd.DataFrame: 모델 최종 입력 데이터
    """

    # sido, gungu = user_df['SIDO'][0], user_df['GUNGU'][0]
    # info_df = info[(info['SIDO'] == sido) & (info['GUNGU'] == gungu)].reset_index(drop=True)
    # info_df.drop(columns=['SIDO', 'GUNGU'], inplace=True)

    # # 사용자 정보를 반복하여 병합
    # user_info_repeated = pd.concat([user_df] * len(info_df), ignore_index=True)
    # final_df = pd.concat([user_info_repeated, info_df], axis=1)
    # final_df.drop_duplicates(subset=['VISIT_AREA_NM'], inplace=True)

    cluster_centers = np.array([
        [126.51347387, 35.62922933],
        [128.58515258, 38.14180494],
        [126.5195361, 33.38336939],
        [128.97343525, 35.15334468],
        [127.04774149, 36.68851356],
        [127.82884433, 34.83825458],
        [126.90789602, 37.52711036],
        [129.00189872, 37.53675183],
        [128.54626983, 36.0416836],
        [127.51159313, 36.3861193],
        [128.07456244, 37.24637136],
        [127.03228769, 35.1726742],
        [126.90050154, 35.9910788],
        [129.34291259, 35.89046611],
        [126.42607313, 34.6591573],
        [126.42859669, 36.62281083]
    ])

    cluster_idx = find_cluster_idx(user_df['LATITUDE'].iloc[0], user_df['LONGITUDE'].iloc[0], cluster_centers)

    filtered_cluster_data = info[info['Cluster'] == cluster_idx].reset_index()
    user_info_repeated = pd.concat([user_df] * len(filtered_cluster_data), ignore_index=True)
    final_df = pd.concat([user_info_repeated, filtered_cluster_data], axis=1)
    final_df.drop_duplicates(subset=['VISIT_AREA_NM'], inplace=True)
    final_df = final_df.drop(['LATITUDE', 'LONGITUDE', 'X_COORD', 'Y_COORD', 'ROAD_NM_ADDR', 'LOTNO_ADDR', 'Cluster', 'Day'], axis=1)
    return final_df


def find_cluster_idx(user_lat: float, user_long: float, cluster_centers: np.ndarray) -> int:
    # 숙소의 좌표를 배열 형태로 변환
    user_location = np.array([user_long, user_lat])

    # 클러스터 중심과 숙소 사이의 거리 계산
    distances = np.linalg.norm(cluster_centers - user_location, axis=1)

    # 가장 작은 거리의 인덱스 찾기
    cluster_idx = np.argmin(distances)

    return cluster_idx


def predict_recommendations(final_df: pd.DataFrame, model_path: str) -> pd.DataFrame:
    """
    모델을 사용하여 여행지 추천 예측 수행
    Args:
        final_df (pd.DataFrame): 모델에 입력할 최종 데이터프레임
        model_path (str): 모델 파일 경로
    Returns:
        pd.DataFrame: 예측 결과가 포함된 데이터프레임
    """
    model = joblib.load(model_path)

    # 모델에 맞는 열 순서로 정렬
    final_df = final_df[model.feature_names_]
    final_df.loc[:, 'y_pred'] = model.predict(final_df)

    return final_df
