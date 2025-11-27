from ultralytics import YOLOv10
import cv2
import os

# --- 설정 (Configuration) ---
# 1. 학습 완료된 Custom 모델 가중치 파일 경로
# 'best.pt' 파일이나, 순수 가중치만 추출된 파일을 사용하세요.
custom_model_path = './runs/detect/yolov10_can_cup_pet_train4/weights/best.pt'
# for debugging
# custom_model_path = './weights/yolov10n.pt' 

# 웹캠 설정: 0은 기본 웹캠을 의미합니다. (다른 웹캠 사용 시 1, 2 등으로 변경 가능)
WEBCAM_INDEX = 0 
CONFIDENCE_THRESHOLD = 0.25 # 탐지 신뢰도 임계값

# --- 1. 모델 로드 ---
try:
    # Ultralytics 기반의 YOLOv10 모델 로드
    # YOLOv10 대신 YOLO를 사용하며, 파일 경로를 전달합니다.
    model = YOLOv10(custom_model_path)
    print(f"✅ Custom 모델 로드 완료: {custom_model_path}")
except Exception as e:
    print(f"❌ 모델 로드 중 오류 발생: {e}")
    exit()

# --- 2. 웹캠 스트림 설정 ---
cap = cv2.VideoCapture(WEBCAM_INDEX)

if not cap.isOpened():
    print(f"❌ 웹캠({WEBCAM_INDEX})을 열 수 없습니다. 카메라 인덱스를 확인하거나 다른 프로그램이 사용 중인지 확인하세요.")
    exit()

print(f"🚀 웹캠 라이브 스트림 시작 (종료: 'q' 키)")

# --- 3. 실시간 추론 루프 ---
try:
    # model.predict()를 사용하여 스트림 방식으로 추론 실행
    # source=WEBCAM_INDEX를 사용하면 OpenCV 없이 Ultralytics가 직접 웹캠을 관리합니다.
    # 하지만 OpenCV 제어의 유연성을 위해 여기서는 cap.read()를 사용합니다.
    
    while True:
        # 프레임 읽기
        ret, frame = cap.read()
        
        if not ret:
            print("웹캠에서 프레임을 읽을 수 없습니다. 스트림을 종료합니다.")
            break

        # YOLOv10 추론 실행
        # stream=True를 사용하면 연속적인 스트림 처리에 최적화됩니다.
        results = model.predict(
            source=frame,
            conf=CONFIDENCE_THRESHOLD,
            verbose=False # 콘솔에 반복적인 로그 출력을 방지
        )
        
        # 결과 프레임 얻기
        # results는 프레임당 하나의 결과 객체를 포함합니다.
        if results and results[0].boxes:
            # results[0].plot()은 검출 결과(바운딩 박스, 라벨)가 그려진 numpy 배열(이미지)을 반환합니다.
            annotated_frame = results[0].plot()
        else:
            # 검출된 객체가 없을 경우 원본 프레임 사용
            annotated_frame = frame

        # 화면에 결과 표시
        cv2.imshow('YOLOv10 Live Detection', annotated_frame)

        # 'q' 키를 누르면 루프 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except Exception as e:
    print(f"실시간 추론 중 예외 발생: {e}")

finally:
    # --- 4. 자원 해제 ---
    cap.release()
    cv2.destroyAllWindows()
    print("라이브 스트림이 종료되었습니다.")