from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
import pickle
import pandas as pd
from utils import process_user_input, filter_and_merge, predict_recommendations
from evaluate import main
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# 백, 프론트 접속 권한 허가 코드
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],  # POST가 여기 포함되어 있는지 확인
    allow_headers=["*"],
)

# 모델 및 데이터 파일 경로
model_path = '../models/catboost_model.pkl'
info_path = '../data/attraction_info.csv'

# 사용자 입력을 위한 Pydantic 모델 정의
class UserInput(BaseModel):
    latitude: float
    longitude: float
    mvmn_nm: str  # 이동수단
    gender: str  # 성별
    age_grp: int  # 나이
    day: int
    travel_styl_1: int
    travel_styl_2: int
    travel_styl_3: int
    travel_styl_5: int
    travel_styl_6: int
    travel_styl_7: int
    travel_styl_8: int
    travel_motive_1: int  # 여행 동기
    rel_cd_categorized: str  # 동행자 정보

class PlaceDTO(BaseModel):
    name: str
    city: str
    address: str
    latitude: float
    longitude: float

class RecommendationDTO(BaseModel):
    day: int
    places: List[PlaceDTO]
    

# 결과를 5개씩 나누어 일차별로 반환하는 함수
def split_recommendations(recommendations, day_count):
    result = []
    # 5개씩 끊어서 리스트 생성
    for i in range(0, len(recommendations), 5):
        day_recommendations = recommendations[i:i+5]
        day = {"day": f"{(i // 5) + 1}일차", "places": day_recommendations}
        result.append(day)
    return result


# API 엔드포인트 구현
@app.post("/ai/trip/short", response_model=List[dict])
async def get_recommendations(user_input: UserInput):
    try:
        # 입력 데이터를 처리하는 부분
        user_df = {
            "LATITUDE": user_input.latitude,
            "LONGITUDE": user_input.longitude,
            "MVMN_NM": user_input.mvmn_nm,
            "GENDER": user_input.gender,
            "AGE_GRP": user_input.age_grp,
            "DAY": user_input.day,
            "TRAVEL_STYL_1": user_input.travel_styl_1,
            "TRAVEL_STYL_2": user_input.travel_styl_2,
            "TRAVEL_STYL_3": user_input.travel_styl_3,
            "TRAVEL_STYL_5": user_input.travel_styl_5,
            "TRAVEL_STYL_6": user_input.travel_styl_6,
            "TRAVEL_STYL_7": user_input.travel_styl_7,
            "TRAVEL_STYL_8": user_input.travel_styl_8,
            "TRAVEL_MOTIVE_1": user_input.travel_motive_1,
            "REL_CD_Categorized": user_input.rel_cd_categorized
            }

        # evaluate.py의 main 함수 호출
        recommendations = main(user_input=user_df, model_path=model_path, info_path=info_path)

        # 데이터 검증 및 NaN, Infinity 값 처리
        for recommendation in recommendations:
            for key, value in recommendation.items():
                if isinstance(value, float) and (pd.isna(value) or value in [float('inf'), float('-inf')]):
                    recommendation[key] = "0.0"  # NaN이나 Inf를 0으로 대체하거나 원하는 값으로 대체

        # 추천 결과를 5개씩 나누어 반환
        split_result = split_recommendations(recommendations, user_input.day)

        # 리스트 형태로 반환
        return split_result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))