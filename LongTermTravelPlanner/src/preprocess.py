import boto3
import pandas as pd
from io import StringIO
from data_processing_utils import impute_missing_values_with_model
import os

def load_data_from_s3(bucket_name: str, file_key: str) -> pd.DataFrame:
    """
    S3 버킷에서 데이터를 로드하는 함수
    
    :param bucket_name: S3 버킷 이름
    :param file_key: S3 파일 경로
    :return: 로드된 데이터프레임
    """
    s3 = boto3.client('s3')
    obj = s3.get_object(Bucket=bucket_name, Key=file_key)
    data = obj['Body'].read().decode('utf-8')
    df = pd.read_csv(StringIO(data))
    
    return df

def load_and_process_data(bucket_name: str) -> pd.DataFrame:
    """
    S3 버킷에서 데이터를 로드하고 전처리합니다.
    
    :param bucket_name: S3 버킷 이름
    :return: 전처리된 데이터프레임
    """
    # S3에서 데이터 로드
    tm_df = load_data_from_s3(bucket_name, '여행객Master.csv')
    travel_df = load_data_from_s3(bucket_name, '여행.csv')
    comp_df = load_data_from_s3(bucket_name, '동반자정보.csv')

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
    def convert_user_id(id):
        return id.split("_")[1]
    
    travel_df['TRAVELER_ID'] = travel_df['TRAVEL_ID'].apply(convert_user_id)
    comp_df['TRAVELER_ID'] = comp_df['TRAVEL_ID'].apply(convert_user_id)

    new_tm_df = tm_df[['TRAVELER_ID', 'TRAVEL_STATUS_ACCOMPANY', 'RESIDENCE_SGG_CD', 'GENDER', 'AGE_GRP',
                       'TRAVEL_NUM', 'TRAVEL_MOTIVE_1', 'TRAVEL_LIKE_SGG_1', 'TRAVEL_LIKE_SGG_2', 'TRAVEL_LIKE_SGG_3']]
    new_travel_df = travel_df[['TRAVELER_ID', 'MVMN_NM']]
    new_comp_df = comp_df[['TRAVELER_ID', 'COMPANION_AGE_GRP', 'REL_CD']]
    
    merged_df = pd.merge(new_tm_df, new_travel_df, on='TRAVELER_ID', how='left')
    merged_df = pd.merge(merged_df, new_comp_df, on='TRAVELER_ID', how='left')
    
    is_solo_travel = merged_df['TRAVEL_STATUS_ACCOMPANY'] == '나홀로 여행'
    merged_df.loc[is_solo_travel, ['COMPANION_AGE_GRP', 'REL_CD']] = merged_df.loc[is_solo_travel, ['COMPANION_AGE_GRP', 'REL_CD']].fillna(-1)
    
    merged_df = merged_df.dropna(subset=['COMPANION_AGE_GRP', 'REL_CD'])
    merged_df = merged_df.drop(columns=['TRAVEL_STATUS_ACCOMPANY'])
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
    # 환경변수에서 S3 버킷 이름을 가져옴
    bucket_name = os.environ['S3_BUCKET_NAME']
    
    # Train 및 Valid 데이터 로드 및 전처리
    train_df = load_and_process_data(bucket_name)
    valid_df = load_and_process_data(bucket_name)
    
    # 결측치 처리
    train_df, valid_df = impute_missing_values_with_model(
        train_df,
        valid_df,
        target_column='MVMN_NM',
        categorical_columns=['RESIDENCE_SGG_CD', 'GENDER', 'TRAVEL_MOTIVE_1', 'COMPANION_AGE_GRP', 'REL_CD']
    )
    
    # 처리된 데이터를 저장
    save_data(train_df, 'processed_train_data.csv')
    save_data(valid_df, 'processed_test_data.csv')

if __name__ == '__main__':
    main()
