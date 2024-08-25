import os
import json
import logging
import argparse
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import tensorflow as tf
from tensorflow.keras import regularizers

import wandb
from data_processing_utils import set_seed  # set_seed 함수 가져오기




# 시드 설정
set_seed(42)

# 명령줄 인자 처리
parser = argparse.ArgumentParser(description='Train a model and save logs.')
parser.add_argument('--log_filename', type=str, default='training.log', help='Log file name')
parser.add_argument('--epochs', type=int, default=50, help='Number of epochs to train')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for the optimizer')
parser.add_argument('--top_k', type=int, default=8, help='Top K predictions for evaluation')
args = parser.parse_args()


# W&B 초기화
wandb.init(project="LongtermTravelRecommender", config={
    "epochs": args.epochs,
    "batch_size": args.batch_size,
    "learning_rate": args.learning_rate,
    "top_k": args.top_k,
})


# logs 폴더 생성 (존재하지 않으면)
log_dir = '../logs'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# 모델 및 하이퍼파라미터 저장 폴더 생성 (존재하지 않으면)
model_dir = '../models'
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# 로그 파일 경로 설정
log_filepath = os.path.join(log_dir, args.log_filename)

# 로그 파일 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filename=log_filepath, filemode='w')
logger = logging.getLogger()

# 데이터 로드
train_data = pd.read_csv('../data/processed/processed_train_data.csv')
test_data = pd.read_csv('../data/processed/processed_test_data.csv')

# 범주형 변수 인코딩
label_encoders = {}

for col in ['GENDER', 'MVMN_NM']:
    le = LabelEncoder()
    train_data[col] = le.fit_transform(train_data[col])
    test_data[col] = le.transform(test_data[col])
    label_encoders[col] = le

# 독립변수와 종속변수 분리
X_train = train_data.drop(columns=['TRAVELER_ID', 'TRAVEL_LIKE_SGG_1', 'TRAVEL_LIKE_SGG_2', 'TRAVEL_LIKE_SGG_3'])
y_train_1 = train_data['TRAVEL_LIKE_SGG_1']
y_train_2 = train_data['TRAVEL_LIKE_SGG_2']
y_train_3 = train_data['TRAVEL_LIKE_SGG_3']

X_test = test_data.drop(columns=['TRAVELER_ID', 'TRAVEL_LIKE_SGG_1', 'TRAVEL_LIKE_SGG_2', 'TRAVEL_LIKE_SGG_3'])
y_test_1 = test_data['TRAVEL_LIKE_SGG_1']
y_test_2 = test_data['TRAVEL_LIKE_SGG_2']
y_test_3 = test_data['TRAVEL_LIKE_SGG_3']

# 학습 데이터와 테스트 데이터를 모두 사용하여 LabelEncoder를 학습
label_encoder = LabelEncoder()
combined_y_train = pd.concat([y_train_1, y_train_2, y_train_3, y_test_1, y_test_2, y_test_3])

# 모든 라벨에 대해 인코더를 학습시킴
label_encoder.fit(combined_y_train)

# 레이블 인코딩
y_train_1 = label_encoder.transform(y_train_1)
y_train_2 = label_encoder.transform(y_train_2)
y_train_3 = label_encoder.transform(y_train_3)

y_test_1 = label_encoder.transform(y_test_1)
y_test_2 = label_encoder.transform(y_test_2)
y_test_3 = label_encoder.transform(y_test_3)

# 고유한 여행지 코드 수 확인
num_classes = len(label_encoder.classes_)
logger.info(f'Number of unique travel destinations (classes): {num_classes}')

# 데이터 표준화
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 모델 생성
input_layer = tf.keras.layers.Input(shape=(X_train.shape[1],))
dense_layer = tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01))(input_layer)
dense_layer = tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01))(dense_layer)
dense_layer = tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01))(dense_layer)

# 각 출력 레이어의 노드 수를 고유한 타깃 클래스 수로 설정
output_1 = tf.keras.layers.Dense(num_classes, activation='softmax', name='output_1')(dense_layer)
output_2 = tf.keras.layers.Dense(num_classes, activation='softmax', name='output_2')(dense_layer)
output_3 = tf.keras.layers.Dense(num_classes, activation='softmax', name='output_3')(dense_layer)

model = tf.keras.Model(inputs=input_layer, outputs=[output_1, output_2, output_3])

# 옵티마이저 설정
optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)

# 각 출력에 대한 손실 함수의 가중치 설정
loss_weights = {'output_1': 0.3, 'output_2': 0.3, 'output_3': 0.3}

# 모델 컴파일 시 손실 함수와 가중치 설정
model.compile(optimizer=optimizer, 
              loss={'output_1': 'sparse_categorical_crossentropy', 
                    'output_2': 'sparse_categorical_crossentropy', 
                    'output_3': 'sparse_categorical_crossentropy'}, 
              loss_weights=loss_weights,
              metrics={'output_1': 'accuracy', 
                       'output_2': 'accuracy', 
                       'output_3': 'accuracy'})

# 각 출력에 대해 별도의 손실 함수와 평가 지표 설정
model.compile(optimizer=optimizer, 
              loss={'output_1': 'sparse_categorical_crossentropy', 
                    'output_2': 'sparse_categorical_crossentropy', 
                    'output_3': 'sparse_categorical_crossentropy'}, 
              metrics={'output_1': 'accuracy', 
                       'output_2': 'accuracy', 
                       'output_3': 'accuracy'})

# 모델 요약 정보 로그로 기록
model.summary(print_fn=lambda x: logger.info(x))

# 학습 로그를 5 에포크마다 기록하기 위한 콜백 설정
class LoggingCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % 5 == 0:
            logger.info(f'Epoch {epoch+1}: {logs}')

logging_callback = LoggingCallback()

class CustomWandbCallback(tf.keras.callbacks.Callback):
    def __init__(self, X_train, y_train_1, y_train_2, y_train_3):
        super(CustomWandbCallback, self).__init__()
        self.X_train = X_train
        self.y_train_1 = y_train_1
        self.y_train_2 = y_train_2
        self.y_train_3 = y_train_3

    def on_epoch_end(self, epoch, logs=None):
        # 10 에포크마다 로그 기록
        if (epoch + 1) % 2 == 0:
            # Gradient norm 계산
            with tf.GradientTape() as tape:
                # 실제 데이터를 사용하여 예측 및 손실 계산
                y_pred = self.model(self.X_train, training=True)
                loss = self.model.compute_loss(self.X_train, [self.y_train_1, self.y_train_2, self.y_train_3], y_pred)

            grads = tape.gradient(loss, self.model.trainable_variables)
            grad_norms = [tf.norm(grad, ord=2).numpy() for grad in grads if grad is not None]
            mean_grad_norm = np.mean(grad_norms)

            # W&B 로깅
            wandb.log({
                'epoch': epoch + 1,
                'loss': logs['loss'],
                'output_1_accuracy': logs['output_1_accuracy'],
                'output_2_accuracy': logs['output_2_accuracy'],
                'output_3_accuracy': logs['output_3_accuracy'],
                'val_loss': logs['val_loss'],
                'val_output_1_accuracy': logs['val_output_1_accuracy'],
                'val_output_2_accuracy': logs['val_output_2_accuracy'],
                'val_output_3_accuracy': logs['val_output_3_accuracy'],
                'grad_norm': mean_grad_norm,
            })

# 콜백 인스턴스 생성 (훈련 데이터를 전달)
custom_wandb_callback = CustomWandbCallback(X_train, y_train_1, y_train_2, y_train_3)





# 모델 훈련
history = model.fit(X_train, [y_train_1, y_train_2, y_train_3], 
                    epochs=args.epochs, 
                    validation_split=0.2, 
                    batch_size=args.batch_size,
                    callbacks=[logging_callback, custom_wandb_callback])


# top_k 값을 명령줄 인자로 받아 설정
top_k = args.top_k

# 예측 수행
preds = model.predict(X_test)

# 각 출력에 대해 상위 k개의 선택으로 변환
top_k_preds_1 = np.argsort(-preds[0], axis=1)[:, :top_k]
top_k_preds_2 = np.argsort(-preds[1], axis=1)[:, :top_k]
top_k_preds_3 = np.argsort(-preds[2], axis=1)[:, :top_k]

# 상위 3개의 선호 여행지에 대한 recall 계산
recalls = []
for i in range(y_test_1.shape[0]):
    # 각 샘플에 대해 모든 예측값을 확률과 함께 결합하고, 상위 8개 선택
    combined_preds = np.concatenate([preds[0][i], preds[1][i], preds[2][i]])  # 모든 예측 확률을 하나로 결합
    combined_indices = np.concatenate([np.arange(num_classes), np.arange(num_classes), np.arange(num_classes)])

    # 확률값에 따라 상위 8개를 선택
    top_k_indices = combined_indices[np.argsort(-combined_preds)][:top_k]

    # 실제 레이블과 비교
    true_labels = [y_test_1[i], y_test_2[i], y_test_3[i]]
    recalls.append(len(set(true_labels) & set(top_k_indices)) / len(true_labels))

mean_recall = np.mean(recalls)


logger.info(f'Mean Recall: {mean_recall}')
print(f'Mean Recall: {mean_recall}')

# 모델 저장
model_save_path = os.path.join(model_dir, 'trained_model.h5')
model.save(model_save_path)
logger.info(f'Model saved to {model_save_path}')

# 모델 아키텍처 저장
model_architecture_path = os.path.join(model_dir, 'model_architecture.json')
with open(model_architecture_path, 'w') as json_file:
    json_file.write(model.to_json())
logger.info(f'Model architecture saved to {model_architecture_path}')

# 하이퍼파라미터와 학습 성능 저장
hyperparams = {
    'epochs': args.epochs,
    'batch_size': args.batch_size,
    'learning_rate': args.learning_rate,
    'optimizer': 'adam',
    'loss': 'sparse_categorical_crossentropy',
    'num_classes': num_classes,
    'mean_recall': mean_recall,
    'top_k': top_k
}

hyperparams_path = os.path.join(model_dir, 'hyperparams.json')
with open(hyperparams_path, 'w') as json_file:
    json.dump(hyperparams, json_file, indent=4)
logger.info(f'Hyperparameters and training performance saved to {hyperparams_path}')

# LabelEncoder 클래스 저장 (각 범주형 변수에 대해)
np.save(os.path.join(model_dir, 'label_classes_gender.npy'), label_encoders['GENDER'].classes_)
np.save(os.path.join(model_dir, 'label_classes_mvmn_nm.npy'), label_encoders['MVMN_NM'].classes_)

# 타깃 라벨 인코더 저장
np.save(os.path.join(model_dir, 'label_classes.npy'), label_encoder.classes_)


# 스케일러 저장
np.save(os.path.join(model_dir, 'scaler_mean.npy'), scaler.mean_)
np.save(os.path.join(model_dir, 'scaler_scale.npy'), scaler.scale_)
