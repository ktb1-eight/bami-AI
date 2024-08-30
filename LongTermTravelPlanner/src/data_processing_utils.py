import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from typing import Optional, List, Union, Tuple
from sklearn.base import BaseEstimator

def set_seed(seed: int):
    """
    모든 라이브러리의 시드를 고정합니다.
    """
    np.random.seed(seed)
    random.seed(seed)
    # scikit-learn의 시드 고정
    from sklearn.utils import check_random_state
    check_random_state(seed)

def impute_missing_values_with_model(
    train_df: pd.DataFrame, 
    valid_df: pd.DataFrame,
    target_column: str, 
    categorical_columns: Optional[List[str]] = None, 
    model: Optional[BaseEstimator] = None,
    random_state: int = 42  # 시드 값을 인자로 받음
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    # 시드 고정
    set_seed(random_state)

    if model is None:
        model = LogisticRegression(random_state=random_state)

    # 결측치가 없는 train 데이터와 valid 데이터 분리
    train_data = train_df[train_df[target_column].notna()]
    test_data_train = train_df[train_df[target_column].isna()]
    
    valid_data = valid_df[valid_df[target_column].notna()]
    test_data_valid = valid_df[valid_df[target_column].isna()]

    # 독립 변수 (features)와 종속 변수 (target) 분리
    X_train = train_data.drop(columns=[target_column])
    y_train = train_data[target_column]
    X_test_train = test_data_train.drop(columns=[target_column])
    
    X_valid = valid_data.drop(columns=[target_column])
    y_valid = valid_data[target_column]
    X_test_valid = test_data_valid.drop(columns=[target_column])

    # 원-핫 인코딩을 위한 ColumnTransformer 설정
    if categorical_columns:
        preprocessor = ColumnTransformer(
            transformers=[
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_columns)
            ],
            remainder='passthrough'  # 나머지 컬럼은 그대로 둠
        )
        model_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])
    else:
        model_pipeline = Pipeline(steps=[('model', model)])

    # 모델 학습
    model_pipeline.fit(X_train, y_train)

    # 결측치 예측
    predicted_values_train = model_pipeline.predict(X_test_train)
    predicted_values_valid = model_pipeline.predict(X_test_valid)

    # 결측치 대체
    train_df.loc[train_df[target_column].isna(), target_column] = predicted_values_train
    valid_df.loc[valid_df[target_column].isna(), target_column] = predicted_values_valid

    return train_df, valid_df
