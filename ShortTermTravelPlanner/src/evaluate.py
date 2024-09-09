from utils import process_user_input, filter_and_merge, predict_recommendations
from typing import Dict, List, Any
import pandas as pd

def main(user_input: Dict[str, Any], model_path: str, info_path: str) -> List[str]:
    """
    사용자의 입력을 받아 여행지 추천을 수행하는 함수.

    Args:
        user_input (Dict[str, Any]): 사용자 입력 데이터
        model_path (str): 모델 파일 경로
        info_path (str): 여행지 정보 파일 경로

    Returns:
        List[str]: 상위 10개의 추천 여행지 리스트
    """
    # 사용자 입력 처리
    user_df = process_user_input(user_input)
    
    # 여행지 정보 로드
    info = pd.read_csv(info_path)
    
    # 여행지 정보 필터링 및 병합
    final_df = filter_and_merge(user_df, info)
    
    # 모델 예측 및 결과 도출
    final_df['VISIT_AREA_TYPE_CD'] = final_df['VISIT_AREA_TYPE_CD'].astype('string')
    final_df['TRAVEL_MOTIVE_1'] = final_df['TRAVEL_MOTIVE_1'].astype('string')
    final_df = predict_recommendations(final_df, model_path)
    
    # 상위 10개 추천지 반환
    top_10_recommendations = final_df.nlargest(10, 'y_pred')['VISIT_AREA_NM']

    # 여행지 정보 필터링 및 매핑
    info = info.rename(columns={'VISIT_AREA_NM': 'name', 'ROAD_NM_ADDR': 'city', 
                                'LOTNO_ADDR': 'address', 'X_COORD': 'longitude', 'Y_COORD': 'latitude'})

    # Vectorized lookup for travel recommendations
    places = info[info['name'].isin(top_10_recommendations)][['name', 'city', 'address', 'longitude', 'latitude']]

    # Convert to list of dictionaries
    return places.to_dict(orient='records')

if __name__ == "__main__":
    user_input = {
        "LATITUDE": 36.75,
        "LONGITUDE": 127.55,
        "MVMN_NM": "자가용",
        "GENDER": "여성",
        "AGE_GRP": 20,
        "Day" : 2,
        "TRAVEL_STYL_1": 1,
        "TRAVEL_STYL_2": 3,
        "TRAVEL_STYL_3": 5,
        "TRAVEL_STYL_5": 4,
        "TRAVEL_STYL_6": 7,
        "TRAVEL_STYL_7": 3,
        "TRAVEL_STYL_8": 4,
        "TRAVEL_MOTIVE_1": 2,
        "REL_CD_Categorized": "혼자"
        }

    
    model_path = '../models/catboost_model.pkl'
    info_path = '../data/attraction_info.csv'
    
    recommendations = main(user_input, model_path, info_path)
    print(f"추천 여행지: {recommendations}")