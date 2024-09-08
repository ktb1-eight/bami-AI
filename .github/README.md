# bami-AI

## 프로젝트 구조

```plaintext

BAMI-ai/
├── LongTerm/
│   ├── data/            # 장기 모델 관련 데이터셋
│   │   ├── raw/         # 원본 데이터
│   │   ├── processed/   # 전처리된 데이터
│   │   └── external/    # 외부 데이터
│   ├── models/          # 장기 모델 파일
│   └── src/             # 장기 모델 학습, 평가, 테스트 스크립트
│       ├── train.py
│       ├── evaluate.py
│       └── utils.py
├── ShortTerm/
│   ├── data/            # 단기 모델 관련 데이터셋
│   │   ├── raw/         # 원본 데이터
│   │   ├── processed/   # 전처리된 데이터
│   │   └── external/    # 외부 데이터
│   ├── models/          # 단기 모델 파일
│   └── src/             # 단기 모델 학습, 평가, 테스트 스크립트
│       ├── train.py
│       ├── evaluate.py
│       └── utils.py
├── visualization/      # 데이터 시각화 관련 파일
│   ├── exploratory/    # 탐색적 데이터 분석 (EDA)
│   └── modeling/       # 모델링 분석 데이터
├── .gitignore          # Git 무시 파일 목록
└── README.md           # 프로젝트 개요 파일
```
