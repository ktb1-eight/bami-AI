import json
import pandas as pd
from catboost import CatBoostRegressor

# 상수 설정
EARLY_STOPPING_ROUNDS = 50
N = 3

# 데이터 로드
train = pd.read_csv('../data/attraction_train_data.csv')
test = pd.read_csv('../data/attraction_test_data.csv')

with open('../models/params.json', 'r') as json_file:
    loaded_params = json.load(json_file)


# NaN 값 제거 및 인덱스 리셋
train.dropna(inplace=True)
train.reset_index(drop=True, inplace=True)

test.dropna(inplace=True)
test.reset_index(drop=True, inplace=True)

# N번 이상 방문한 곳으로 필터링
visit_counts = train['VISIT_AREA_NM'].value_counts()
frequent_places = visit_counts[visit_counts >= N].index
train = train[train['VISIT_AREA_NM'].isin(
    frequent_places)].reset_index(drop=True)

# 학습에 필요 없는 feature 제거
drop_features = ['TRAVELER_ID', 'REVISIT_INTENTION',
                 'RCMDTN_INTENTION', 'RESIDENCE_TIME_MIN', 'REVISIT_YN',
                 'TRAVEL_COMPANIONS_NUM']
train.drop(columns=drop_features, inplace=True)

# 데이터 타입 변경
cat_features = ['VISIT_AREA_NM', 'SIDO', 'GUNGU', 'VISIT_AREA_TYPE_CD',
                'GENDER', 'MVMN_NM', 'TRAVEL_MOTIVE_1', 'REL_CD_Categorized']

train[cat_features] = train[cat_features].astype(str)

# 학습 및 타겟 데이터 설정
X_train = train.drop(columns=['DGSTFN', 'TRAVEL_ID'])
y_train = train['DGSTFN']

model = CatBoostRegressor(**loaded_params, cat_features=['VISIT_AREA_NM', 'SIDO', 'GUNGU', 'VISIT_AREA_TYPE_CD', 
                                                     'MVMN_NM', 'GENDER', 'REL_CD_Categorized', 'TRAVEL_MOTIVE_1'],
                          random_state=42, early_stopping_rounds=EARLY_STOPPING_ROUNDS)

model.fit(X_train, y_train)

print(model.best_score_)