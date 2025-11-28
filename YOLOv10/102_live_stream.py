from ultralytics import YOLOv10
import cv2
import os

# --- 1. 설정 (Configuration) ---
# 'yolov10n.pt' 파일이 현재 경로의 './weights/' 폴더에 있다고 가정합니다.
model_name = './weights/yolov10n.pt'
try:
    # 💡 모델 로드
    model = YOLOv10(model_name)
    print(f"✅ 모델 로드 완료: {model_name}")
except FileNotFoundError:
    print(f"❌ 오류: 모델 파일 '{model_name}'을 찾을 수 없습니다. 경로를 확인하거나 수동으로 다운로드하세요.")
    exit()

# 💡 클래스 ID에 따른 고유한 색상을 생성하는 함수 정의
def get_color_by_class(cls_id):
    """클래스 ID에 따라 고유한 BGR 색상 튜플을 반환합니다."""
    # [B, G, R] 순서로 20개의 색상을 미리 정의합니다.
    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 0, 255), 
        (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0), (0, 128, 128), (128, 0, 128),
        (255, 128, 0), (255, 0, 128), (0, 255, 128), (128, 255, 0), (128, 0, 255), (0, 128, 255),
        (255, 255, 128), (128, 255, 255)
    ]
    return colors[cls_id % len(colors)]

# --- 2. 웹캠 캡처 객체 초기화 ---
# 0번은 보통 기본 웹캠을 의미합니다. 여러 대의 카메라가 연결된 경우 1, 2 등으로 변경하세요.
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ 오류: 카메라를 열 수 없습니다. 카메라가 연결되어 있는지 또는 인덱스(0)가 올바른지 확인하세요.")
    exit()

# --- 3. 실시간 프레임 처리 루프 ---
print("🚀 웹캠 실시간 객체 검출을 시작합니다. 'q' 키를 누르면 종료됩니다.")
# 
while cap.isOpened():
    # 프레임 읽기
    success, frame = cap.read()
    
    if not success:
        print("프레임을 받을 수 없습니다 (스트림 끝?). 종료합니다.")
        break
    
    # 추론 실행 (프레임마다 실행)
    results = model.predict(
        source=frame,  # 현재 프레임을 직접 전달
        conf=0.25,
        verbose=False,
        stream=True  # 실시간 스트림 처리를 최적화
    )

    # 결과 시각화
    for result in results: # stream=True인 경우, results는 제너레이터입니다.
        
        # 검출된 결과에서 바운딩 박스, 클래스 ID, 신뢰도 추출
        boxes = result.boxes.xyxy.cpu().numpy()  # 바운딩 박스 좌표
        classes = result.boxes.cls.cpu().numpy().astype(int)  # 클래스 ID
        confidences = result.boxes.conf.cpu().numpy()  # 신뢰도

        # 시각화 설정
        thickness = 2
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7

        # 현재 프레임에 바운딩 박스 및 레이블 그리기
        for box, cls_id, conf in zip(boxes, classes, confidences):
            x1, y1, x2, y2 = map(int, box)
            class_name = model.names[cls_id]

            # 색상 설정 (클래스 ID 기반으로 고유한 색상 생성)
            color = get_color_by_class(cls_id)
            
            # 바운딩 박스 그리기
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

            # 레이블 텍스트 생성: "클래스 이름 (신뢰도%)"
            label = f"{class_name} ({conf:.2f})"
            
            # 텍스트 크기 계산
            (w, h), _ = cv2.getTextSize(label, font, font_scale, thickness)
            
            # 텍스트 배경을 위한 사각형 그리기
            cv2.rectangle(frame, (x1, y1 - h - 5), (x1 + w, y1), color, -1)
            
            # 텍스트 그리기
            cv2.putText(frame, label, (x1, y1 - 5), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
            
            # 실시간 출력에서는 print를 최소화합니다.
            # print(f"  - 클래스: {class_name}, 신뢰도: {conf:.2f}")

    # 화면에 프레임 표시
    cv2.imshow('YOLOv10 Live Detection', frame)

    # 'q' 키를 누르면 루프 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- 4. 종료 및 자원 해제 ---
cap.release()
cv2.destroyAllWindows()
print("\n🎉 실시간 검출을 종료합니다.")