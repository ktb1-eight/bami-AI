import pandas as pd
import os
from sklearn.linear_model import LogisticRegression
from data_processing_utils import impute_missing_values_with_model

def load_and_process_data(data_dir: str) -> pd.DataFrame:
    """
    데이터 폴더에서 데이터를 로드하고 전처리합니다.
    
    :param data_dir: 데이터가 포함된 디렉토리 경로
    :return: 전처리된 데이터프레임
    """
    # 파일 경로 설정 및 데이터 로드
    tm_df = pd.read_csv(os.path.join(data_dir, '여행객Master.csv'))
    travel_df = pd.read_csv(os.path.join(data_dir, '여행.csv'))
    comp_df = pd.read_csv(os.path.join(data_dir, '동반자정보.csv'))

    # 데이터 병합 및 전처리
    merged_df = preprocess_data(tm_df, travel_df, comp_df)
    
    return merged_df

def preprocess_data(tm_df, travel_df, comp_df):
    """
    데이터를 전처리하고 병합하는 함수.
    
    :param tm_df: 여행객Master 데이터프레임
    :param travel_df: 여행 데이터프레임
    :param comp_df: 동반자정보 데이터프레임
    :return: 병합되고 전처리된 데이터프레임
    """
    # 사용자 ID 변환 함수
    def convert_user_id(id):
        return id.split("_")[1]
    
    # TRAVELER_ID 컬럼 생성
    travel_df['TRAVELER_ID'] = travel_df['TRAVEL_ID'].apply(convert_user_id)
    comp_df['TRAVELER_ID'] = comp_df['TRAVEL_ID'].apply(convert_user_id)

    # 필요한 컬럼만 선택
    new_tm_df = tm_df[['TRAVELER_ID', 'TRAVEL_STATUS_ACCOMPANY', 'RESIDENCE_SGG_CD', 'GENDER', 'AGE_GRP',
                       'TRAVEL_NUM', 'TRAVEL_MOTIVE_1', 'TRAVEL_LIKE_SGG_1', 'TRAVEL_LIKE_SGG_2', 'TRAVEL_LIKE_SGG_3']]
    new_travel_df = travel_df[['TRAVELER_ID', 'MVMN_NM']]
    new_comp_df = comp_df[['TRAVELER_ID', 'COMPANION_AGE_GRP', 'REL_CD']]
    
    # 세 데이터프레임 병합
    merged_df = pd.merge(new_tm_df, new_travel_df, on='TRAVELER_ID', how='left')
    merged_df = pd.merge(merged_df, new_comp_df, on='TRAVELER_ID', how='left')
    
    # '나홀로 여행'인 경우 COMPANION_AGE_GRP와 REL_CD의 결측치를 -1로 채우기
    is_solo_travel = merged_df['TRAVEL_STATUS_ACCOMPANY'] == '나홀로 여행'
    merged_df.loc[is_solo_travel, ['COMPANION_AGE_GRP', 'REL_CD']] = merged_df.loc[is_solo_travel, ['COMPANION_AGE_GRP', 'REL_CD']].fillna(-1)

    # '나홀로 여행'이 아닌 경우 결측치가 있는 행 드랍
    merged_df = merged_df.dropna(subset=['COMPANION_AGE_GRP', 'REL_CD'])

    # 불필요한 컬럼 삭제
    merged_df = merged_df.drop(columns=['TRAVEL_STATUS_ACCOMPANY'])

    # TRAVELER_ID를 인덱스로 설정
    merged_df.set_index('TRAVELER_ID', inplace=True)
    
    return merged_df

def save_data(df: pd.DataFrame, output_path: str):
    """
    데이터프레임을 CSV 파일로 저장합니다.
    
    :param df: 데이터프레임
    :param output_path: 저장할 파일 경로
    """
    df.to_csv(output_path)

def main():
    # Train 데이터 로드 및 전처리
    train_df = load_and_process_data('../data/raw/train')
    
    # Valid 데이터 로드 및 전처리
    valid_df = load_and_process_data('../data/raw/valid')
    
    # Train 데이터의 결측치를 채우고, Valid 데이터의 결측치도 채움
    train_df, valid_df = impute_missing_values_with_model(
        train_df,
        valid_df,
        target_column='MVMN_NM',
        categorical_columns=['RESIDENCE_SGG_CD', 'GENDER', 'TRAVEL_MOTIVE_1', 'COMPANION_AGE_GRP', 'REL_CD']
    )
    
    # 채워진 데이터를 CSV 파일로 저장
    save_data(train_df, 'processed_train_data.csv')
    save_data(valid_df, 'processed_test_data.csv')

if __name__ == '__main__':
    main()
