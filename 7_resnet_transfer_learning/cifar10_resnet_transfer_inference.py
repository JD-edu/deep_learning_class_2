import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input

# 1. 저장된 모델 불러오기
model = tf.keras.models.load_model('resnet_transfer_model.h5')

# 2. 클래스 라벨 설정 (CIFAR10 기준 또는 커스텀 데이터 라벨)
# CIFAR10: ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
# 커스텀 3종(예시): ['can', 'cup', 'pet']
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# 3. 카메라 설정 (0번은 기본 웹캠)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("카메라를 열 수 없습니다.")
    exit()

print("실시간 추론을 시작합니다. 'q'를 누르면 종료합니다.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # --- 이미지 전처리 ---
    # 1) BGR(OpenCV)을 RGB로 변환
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # 2) 모델의 입력 사이즈로 리사이징 (학습 시 설정한 사이즈와 동일해야 함)
    img = cv2.resize(img, (32, 32)) 
    # 3) 배치 차원 추가 (1, 32, 32, 3)
    img_array = np.expand_dims(img, axis=0)
    # 4) ResNet50 전처리 함수 적용
    img_array = preprocess_input(img_array)

    # --- 추론 실행 ---
    predictions = model.predict(img_array, verbose=0)
    score = tf.nn.softmax(predictions[0]) # 확률 점수 계산
    class_idx = np.argmax(predictions[0]) # 가장 높은 확률의 인덱스
    label = class_names[class_idx]
    confidence = 100 * np.max(predictions[0])

    # --- 결과 화면 표시 ---
    text = f"{label} ({confidence:.2f}%)"
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Real-time Inference', frame)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()