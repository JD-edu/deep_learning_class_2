from ultralytics import YOLOv10
import os

# --- 설정 (Configuration) ---
# 1. 학습 완료된 Custom 모델 가중치 파일 경로
# 'best.pt'는 2단계에서 학습 완료 후 저장된 파일입니다.
custom_model_path = './runs/detect/yolov10_can_cup_pet_train4/weights/best.pt'
# for debugging 
#custom_model_path = './weights/yolov10n.pt'
# 2. 테스트할 이미지 파일 경로
image_path = './custom_images/cans/cans (1).jpg' 

if not os.path.exists(custom_model_path):
    print(f"❌ 오류: Custom 모델 파일 '{custom_model_path}'을 찾을 수 없습니다. 경로를 확인하세요.")
    exit()
if not os.path.exists(image_path):
    print(f"❌ 오류: 이미지 파일 '{image_path}'을 찾을 수 없습니다. 경로를 확인하세요.")
    exit()

# --- 2. 모델 로드 및 추론 실행 ---
try:
    model = YOLOv10(custom_model_path)
    print(f"✅ Custom 모델 로드 완료: {custom_model_path}")
except Exception as e:
    print(f"❌ 모델 로드 중 오류 발생: {e}")
    exit()

print(f"🚀 이미지 추론 시작: {image_path}")

# 추론 실행
results = model.predict(
    source=image_path,
    conf=0.25,      # 탐지 신뢰도 임계값
    save=True,      # 결과 이미지를 runs/detect/predict 폴더에 저장
    show=True,      # 결과를 화면에 표시 (OpenCV 필요)
    name='custom_image_result' # 결과 저장 폴더 이름 지정
)

# --- 3. 결과 출력 ---
if results:
    # results[0]은 첫 번째 이미지의 결과
    num_detections = len(results[0].boxes)
    print(f"\n✅ 검출된 총 객체 수: {num_detections}")
    print(f"🎉 결과 이미지는 'runs/detect/custom_image_result' 폴더에 저장되었습니다.")
else:
    print("❌ 검출된 객체가 없습니다.")