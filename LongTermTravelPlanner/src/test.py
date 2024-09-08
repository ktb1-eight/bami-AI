import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os

# 모델 및 인코더 파일 경로 설정
model_dir = '../models'
model_path = os.path.join(model_dir, 'trained_model.h5')

# 모델 로드
model = tf.keras.models.load_model(model_path)

# 인코더 로드
label_encoder_gender = LabelEncoder()
label_encoder_gender.classes_ = np.load(os.path.join(model_dir, 'label_classes_gender.npy'), allow_pickle=True)

label_encoder_mvmn_nm = LabelEncoder()
label_encoder_mvmn_nm.classes_ = np.load(os.path.join(model_dir, 'label_classes_mvmn_nm.npy'), allow_pickle=True)

# 스케일러 로드
scaler = StandardScaler()
scaler.mean_ = np.load(os.path.join(model_dir, 'scaler_mean.npy'), allow_pickle=True)
scaler.scale_ = np.load(os.path.join(model_dir, 'scaler_scale.npy'), allow_pickle=True)

# 여행지 예측 라벨 인코더 로드
label_encoder_travel = LabelEncoder()
label_encoder_travel.classes_ = np.load(os.path.join(model_dir, 'label_classes.npy'), allow_pickle=True)

# SGG_CD 데이터 로드
sgg_cd_df = pd.read_csv('../config/SGG_CD.csv')
# f000415,41,여,40,1,8,26350,47130,46130,대중교통 등,-1.0,-1.0
# f000363,50,남,50,1,1,11110,50130,28710,자가용,4.0,1.0
# 예측할 임의의 유저 데이터
user_data = {
    'RESIDENCE_SGG_CD': 50,
    'GENDER': '남',
    'AGE_GRP': 50,
    'TRAVEL_NUM': 1,
    'TRAVEL_MOTIVE_1': 1,
    'MVMN_NM': '자가용',
    'COMPANION_AGE_GRP': 4.0,
    'REL_CD': 1.0
}

# 데이터프레임으로 변환
user_df = pd.DataFrame([user_data])

# 범주형 데이터 인코딩 (GENDER와 MVMN_NM에 대해 각각)
user_df['GENDER'] = label_encoder_gender.transform(user_df['GENDER'])
user_df['MVMN_NM'] = label_encoder_mvmn_nm.transform(user_df['MVMN_NM'])

# 데이터 표준화
user_scaled = scaler.transform(user_df)

# 예측 수행
preds = model.predict(user_scaled)

# 각 출력에 대해 상위 8개의 선택과 확률을 가져옴
top_k = 8
top_k_indices = np.argsort(-preds[0], axis=1)[:, :top_k]
top_k_probs = -np.sort(-preds[0], axis=1)[:, :top_k]

# 예측된 인덱스를 원래 라벨로 변환
top_k_labels = label_encoder_travel.inverse_transform(top_k_indices[0])
print(top_k_labels)
# 결과 출력: 1순위 여행지 예측의 top 8과 그 확률, 그리고 SIDO_NM과 SGG_NM 정보
print("1순위 여행지 예측 (Top 8) 및 상세 정보:")
for i in range(top_k):
    sgg_code = top_k_labels[i]
    sgg_info = sgg_cd_df[sgg_cd_df['SGG_CD'] == sgg_code]
    sido_nm = sgg_info['SIDO_NM'].values[0] if not sgg_info.empty else "정보 없음"
    sgg_nm = sgg_info['SGG_NM'].values[0] if not sgg_info.empty else "정보 없음"
    print(f"{i+1}순위: {sgg_code} (확률: {top_k_probs[0][i]:.4f}) - {sido_nm} {sgg_nm}")
