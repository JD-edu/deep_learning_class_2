# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import ViTForImageClassification, ViTImageProcessor
import time
import os
import numpy as np
import matplotlib.pyplot as plt

# 1. 저장된 모델 경로 설정 (📌 학습 후 저장된 폴더 이름과 일치해야 합니다)
MODEL_PATH = "./final_custom_vit_model"

# 2. 추론할 이미지 파일 경로 설정 (📌 실제 이미지 파일로 변경하세요)
IMAGE_FILE_PATH = "./cans.jpg"

# -----------------------------------------------------
# 모델 및 프로세서 로드
# -----------------------------------------------------
try:
    if not os.path.isdir(MODEL_PATH):
        raise FileNotFoundError(f"모델 폴더를 찾을 수 없습니다: {MODEL_PATH}")

    # processor는 저장된 설정(Resize, Normalize 등)을 로드합니다.
    processor = ViTImageProcessor.from_pretrained(MODEL_PATH)
    
    # ViT 모델을 로드할 때 'output_attentions=True' 옵션을 설정하여 Attention Weights를 반환하도록 합니다.
    model = ViTForImageClassification.from_pretrained(MODEL_PATH, output_attentions=True)

    # GPU 사용 가능 여부 확인 후 디바이스 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval() # 추론 모드로 설정

    print(f"✅ 모델 로드 완료. 현재 Device: {device}")

except FileNotFoundError as e:
    print(f"❌ 오류: {e}")
    print("모델 저장 폴더와 파일(pytorch_model.bin, config.json 등)의 위치를 확인해주세요.")
    exit()
except Exception as e:
    print(f"❌ 모델 로드 중 예기치 않은 오류 발생: {e}")
    exit()

# -----------------------------------------------------
# Attention Map 시각화 함수
# -----------------------------------------------------

def visualize_attention(image, attentions):
    """
    Attention Weights를 사용하여 Heatmap을 생성하고 원본 이미지 위에 오버레이합니다.
    
    Args:
        image (PIL.Image): 원본 이미지.
        attentions (tuple of torch.Tensor): 모델 출력의 attentions 튜플.
    """
    # 1. [CLS] 토큰에 대한 Attention Weights 추출 및 평균 계산
    # attentions shape: (num_layers, batch_size, num_heads, num_tokens, num_tokens)
    
    # 마지막 레이어의 Attention Weights 사용 (가장 추상적인 특징을 담고 있음)
    # shape: (1, num_heads, num_tokens, num_tokens)
    last_layer_attentions = attentions[-1].detach().cpu() 
    
    # [CLS] 토큰 (인덱스 0)이 다른 패치 토큰 (인덱스 1부터)에 주는 Attention 추출
    # shape: (1, num_heads, num_tokens) -> [:, :, 0, 1:]
    cls_attention = last_layer_attentions[0, :, 0, 1:]
    
    # 모든 헤드에 대해 평균을 취하여 최종 Attention Map 생성
    # shape: (num_tokens) -> 256
    mean_attention = cls_attention.mean(dim=0).numpy()

    # 2. Attention Map을 2D 형태로 변환 및 리사이즈
    # ViT는 224x224 이미지를 16x16 패치로 나누면 14x14 = 196 토큰 (+1 CLS)이 생성됨
    # 모델의 config에서 패치 크기를 추출하는 것이 가장 정확하나, 여기서는 일반적인 경우를 가정
    patch_size = model.config.patch_size
    img_size = processor.size["height"] # 224x224
    map_dim = img_size // patch_size  # 224 / 16 = 14
    
    # 1차원 Attention 가중치 (196)를 2차원 맵 (14x14)으로 변환
    attention_map = mean_attention.reshape(map_dim, map_dim)

    # 원본 이미지 크기로 맵을 업스케일링 (PIL의 nearest 대신 BILINEAR 사용)
    attention_map_resized = Image.fromarray(attention_map).resize(image.size, Image.Resampling.BILINEAR)
    attention_map_np = np.array(attention_map_resized)
    
    # 3. 시각화
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    
    # 원본 이미지
    ax[0].imshow(image)
    ax[0].set_title("Original Image")
    ax[0].axis('off')

    # Heatmap 오버레이
    ax[1].imshow(image) 
    # 'jet' 컬러맵 사용 및 투명도(alpha) 조절
    im = ax[1].imshow(attention_map_np, alpha=0.6, cmap='jet')
    ax[1].set_title("Attention Heatmap (Model Focus)")
    ax[1].axis('off')
    
    # 컬러바 추가
    fig.colorbar(im, ax=ax[1], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.show()

# -----------------------------------------------------
# 추론 실행
# -----------------------------------------------------
try:
    # 이미지 로드 (모델 전처리 요구사항에 맞춰 RGB로 변환)
    start_time = time.time()
    image = Image.open(IMAGE_FILE_PATH).convert("RGB")
    original_image = image.copy() # 시각화를 위해 원본 이미지 복사

    # 4. 이미지 전처리 및 텐서 생성
    inputs = processor(images=image, return_tensors="pt").to(device)

    # 5. 예측 (Prediction) 수행 (output_attentions=True 이므로 attentions도 반환됨)
    with torch.no_grad():
        # output은 logits, hidden_states, attentions 등을 포함하는 튜플/객체
        outputs = model(**inputs)

    # 6. 결과 해석
    probabilities = F.softmax(outputs.logits, dim=-1)
    predicted_prob = probabilities.max().item()
    predicted_class_idx = probabilities.argmax(-1).item()

    # 7. 클래스 이름으로 변환
    predicted_label = model.config.id2label[predicted_class_idx]

    end_time = time.time()

    print("\n--- 추론 결과 ---")
    print(f"소요 시간: {end_time - start_time:.4f} 초")
    print(f"예측된 클래스: **{predicted_label}**")
    print(f"예측 확률: {predicted_prob:.4f}")
    
    # 8. Attention Map 시각화 실행
    print("\n🎨 Attention Heatmap 생성 중...")
    visualize_attention(original_image, outputs.attentions)

except FileNotFoundError:
    print(f"오류: 이미지 파일 '{IMAGE_FILE_PATH}'을(를) 찾을 수 없습니다. 경로를 다시 확인해주세요.")
except Exception as e:
    print(f"추론 과정 중 오류 발생: {e}")