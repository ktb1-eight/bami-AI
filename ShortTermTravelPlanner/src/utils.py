from typing import Dict, Any
import joblib
import pandas as pd

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
        'SIDO', 'GUNGU', 'MVMN_NM', 'GENDER', 'REL_CD_Categorized', 'TRAVEL_MOTIVE_1'
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

    sido, gungu = user_df['SIDO'][0], user_df['GUNGU'][0]
    info_df = info[(info['SIDO'] == sido) & (info['GUNGU'] == gungu)].reset_index(drop=True)
    info_df.drop(columns=['SIDO', 'GUNGU'], inplace=True)
    
    # 사용자 정보를 반복하여 병합
    user_info_repeated = pd.concat([user_df] * len(info_df), ignore_index=True)
    final_df = pd.concat([user_info_repeated, info_df], axis=1)
    final_df.drop_duplicates(subset=['VISIT_AREA_NM'], inplace=True)
    print(final_df)

    return final_df

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