from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import pickle
import pandas as pd
# 필요한 함수들을 import (process_user_input, filter_and_merge, predict_recommendations)
from utils import process_user_input, filter_and_merge, predict_recommendations
# evalutae.py 의 main 함수 가져오기
from evaluate import main


app = FastAPI()

# 모델 및 데이터 파일 경로
model_path = '/카카오테크 부트캠프/Team_Project1/bami-AI/ShortTermTravelPlanner/models/catboost_model.pkl'
info_path = '/카카오테크 부트캠프/Team_Project1/bami-AI/ShortTermTravelPlanner/data/attraction_info.csv'

# 사용자 입력을 위한 Pydantic 모델 정의
class UserInput(BaseModel):
    LATITUDE: float
    LONGITUDE: float
    MVMN_NM: str # 이동수단
    GENDER: str # 성별
    AGE_GRP: int # 나이
    DAY : int
    TRAVEL_STYL_1: int
    TRAVEL_STYL_2: int
    TRAVEL_STYL_3: int
    TRAVEL_STYL_5: int
    TRAVEL_STYL_6: int
    TRAVEL_STYL_7: int
    TRAVEL_STYL_8: int
    TRAVEL_MOTIVE_1: int # 여행 동기
    REL_CD_Categorized: str # 동행자 정보

# API 엔드포인트 구현
@app.post("/ai/trip/short", response_model = List[str]) # List[str] : API가 반환할 응답의 타입
def get_recommendations(user_input: UserInput):
    try:
        # 입력 데이터를 처리하는 부분
        user_df = user_input.dict()
        
        print("\n")
        
        # evaluate.py의 main 함수 호출
        recommendations = main(user_input=user_df, 
                               model_path=model_path, 
                               info_path=info_path, 
                               latitude=user_input.LATITUDE, 
                               longitude=user_input.LONGITUDE)
        print(recommendations)
        
        return recommendations
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))