from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware  # CORS 처리를 위한 모듈
from pydantic import BaseModel, Field
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os

# FastAPI 인스턴스 생성
app = FastAPI()

# CORS 설정 추가
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 모든 도메인에서의 요청을 허용 (원하는 도메인을 지정할 수 있음)
    allow_credentials=True,
    allow_methods=["*"],  # 모든 HTTP 메서드 허용 (GET, POST, PUT 등)
    allow_headers=["*"],  # 모든 헤더 허용
)

# 모델 및 인코더 파일 경로 설정
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(BASE_DIR, 'models')

# 모델 로드
model = tf.keras.models.load_model(os.path.join(model_path,'trained_model.h5'))

# 인코더 로드
label_encoder_gender = LabelEncoder()
label_encoder_gender.classes_ = np.load(os.path.join(model_path, 'label_classes_gender.npy'), allow_pickle=True)

label_encoder_mvmn_nm = LabelEncoder()
label_encoder_mvmn_nm.classes_ = np.load(os.path.join(model_path, 'label_classes_mvmn_nm.npy'), allow_pickle=True)

# 스케일러 로드
scaler = StandardScaler()
scaler.mean_ = np.load(os.path.join(model_path, 'scaler_mean.npy'), allow_pickle=True)
scaler.scale_ = np.load(os.path.join(model_path, 'scaler_scale.npy'), allow_pickle=True)

# 여행지 예측 라벨 인코더 로드
label_encoder_travel = LabelEncoder()
label_encoder_travel.classes_ = np.load(os.path.join(model_path, 'label_classes.npy'), allow_pickle=True)

# SGG_CD 데이터 로드
sgg_cd_df = pd.read_csv(os.path.join(BASE_DIR,'config','SGG_CD.csv'))

# 입력 데이터 스키마 정의 (대문자 키를 허용하지 않도록 설정)
class UserInput(BaseModel):
    residence_sgg_cd: int
    gender: str
    age_grp: int
    travel_num: int
    travel_motive_1: int
    mvmn_nm: str
    companion_age_grp: float
    rel_cd: float

# POST 요청을 처리하는 엔드포인트
@app.post("/api/predict")
async def predict_travel_destination(user_input: UserInput):
    try:
        # 입력 데이터를 DataFrame으로 변환
        user_df = pd.DataFrame([user_input.dict()])

        # 범주형 데이터 인코딩
        user_df['gender'] = label_encoder_gender.transform(user_df['gender'])
        user_df['mvmn_nm'] = label_encoder_mvmn_nm.transform(user_df['mvmn_nm'])

        # 데이터 표준화
        user_scaled = scaler.transform(user_df)

        # 예측 수행
        preds = model.predict(user_scaled)

        # 상위 8개 선택과 확률 추출
        top_k = 8
        top_k_indices = np.argsort(-preds[0], axis=1)[:, :top_k]
        top_k_probs = -np.sort(-preds[0], axis=1)[:, :top_k]

        # 예측된 인덱스를 원래 라벨로 변환
        top_k_labels = label_encoder_travel.inverse_transform(top_k_indices[0])

        # 결과 구성
        results = []
        for i in range(top_k):
            sgg_code = top_k_labels[i]
            sgg_info = sgg_cd_df[sgg_cd_df['SGG_CD'] == sgg_code]
            sido_nm = sgg_info['SIDO_NM'].values[0] if not sgg_info.empty else "정보 없음"
            sgg_nm = sgg_info['SGG_NM'].values[0] if not sgg_info.empty else "정보 없음"
            
            # region의 값만 어펜드
            results.append(f"{sido_nm} {sgg_nm}")

        return results

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
