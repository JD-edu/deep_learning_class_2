import torch
from PIL import Image
from transformers import ViTForImageClassification, ViTImageProcessor
import cv2  # 웹캠 사용을 위한 OpenCV 라이브러리
import numpy as np # OpenCV와 PIL 이미지 변환을 위한 NumPy

# 1. 모델 및 장치 설정 (기존 코드와 동일)
# 모델이 저장된 폴더 경로 (현재 디렉토리)
MODEL_PATH = "./" 

try:
    processor = ViTImageProcessor.from_pretrained(MODEL_PATH)
    model = ViTForImageClassification.from_pretrained(MODEL_PATH)
    # CPU 환경을 기본으로 사용합니다.
    device = torch.device("cpu") 
    model.to(device)
    model.eval() # 추론 모드로 설정
    print(f"✅ 모델 로드 완료. 현재 Device: {device}")
except Exception as e:
    print(f"❌ 모델 로드 중 오류 발생: {e}")
    print("폴더 경로와 파일 존재 여부를 확인해주세요.")
    exit()

# 2. 웹캠 초기화
# 0은 보통 기본 웹캠을 의미합니다. 다른 카메라를 사용하려면 숫자를 변경하세요 (예: 1, 2)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ 오류: 웹캠을 열 수 없습니다. 카메라 연결을 확인하세요.")
    exit()

print("\n--- 라이브 스트림 시작 (ESC 또는 'q' 키를 누르면 종료) ---")

# 3. 라이브 스트림 루프
while True:
    # 프레임 읽기
    ret, frame = cap.read()
    
    if not ret:
        print("프레임을 읽을 수 없습니다.")
        break
    
    # --- 4. ViT 모델 입력 형식으로 변환 ---
    # OpenCV는 BGR (Blue-Green-Red) 순서, ViT 프로세서는 RGB 순서를 기대합니다.
    # 1. BGR -> RGB 변환 (NumPy 배열 상태)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # 2. NumPy 배열을 PIL Image 객체로 변환 (ViT 프로세서 입력에 적합)
    image = Image.fromarray(frame_rgb)
    
    # 3. ViT 프로세서로 전처리 및 텐서 생성
    inputs = processor(images=image, return_tensors="pt").to(device)

    # 5. 추론 (Prediction) 수행
    with torch.no_grad():
        outputs = model(**inputs)

    # 6. 결과 해석
    logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()
    
    # 7. 클래스 이름으로 변환
    predicted_label = model.config.id2label[predicted_class_idx]
    
    # --- 8. 결과 화면에 표시 ---
    # 예측 결과를 OpenCV 프레임에 오버레이
    text = f"Predicted: {predicted_label}"
    
    # (프레임, 텍스트, 시작 위치, 폰트, 크기, 색상, 두께)
    cv2.putText(frame, 
                text, 
                (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                1, 
                (0, 255, 0), # BGR 색상: 초록색
                2, 
                cv2.LINE_AA)
    
    # 화면에 프레임 표시
    cv2.imshow('ViT Live Classification', frame)
    
    # 9. 종료 조건
    # 'q' 키 또는 ESC 키를 누르면 루프 종료
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27: 
        break

# 10. 자원 해제
cap.release()
cv2.destroyAllWindows()
print("라이브 스트림 종료 및 자원 해제 완료.")