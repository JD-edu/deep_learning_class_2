# -*- coding: utf-8 -*-

import torch
import numpy as np
from PIL import Image
from transformers import ViTForImageClassification, ViTImageProcessor, TrainingArguments, Trainer
from torchvision.transforms import (Compose, Normalize, RandomHorizontalFlip, RandomResizedCrop, ToTensor, Resize, CenterCrop)
from torch.utils.data import DataLoader
from datasets import load_dataset
from sklearn.metrics import accuracy_score
import os

# 양자화를 위한 라이브러리 추가
from accelerate import Accelerator
from accelerate.utils import find_executable_batch_size 
# (Hugging Face의 모델 로드 및 추론 최적화 기능 사용)

# ... (기존 데이터 로드 및 전처리 코드 유지) ...

CUSTOM_DATA_ROOT = "./images" # 'can', 'pet', 'cup' 폴더가 포함된 상위 폴더

from datasets import load_dataset
raw_datasets = load_dataset('imagefolder', data_dir=CUSTOM_DATA_ROOT)

splits = raw_datasets['train'].train_test_split(test_size=0.1, seed=42)
train_ds = splits['train']
val_ds = splits['test']

labels = train_ds.features['label'].names
id2label = {i: label for i, label in enumerate(labels)}
label2id = {label: i for i, label in enumerate(labels)}
num_labels = len(labels)

print(f" 로드된 클래스: {labels}")
print(f" 총 훈련 데이터 개수: {len(train_ds)}")

model_name_or_path = 'google/vit-base-patch16-224-in21k'
processor = ViTImageProcessor.from_pretrained(model_name_or_path)

image_mean, image_std = processor.image_mean, processor.image_std
size = processor.size["height"]

normalize = Normalize(mean=image_mean, std=image_std)
_train_transforms = Compose(
    [RandomResizedCrop(size), RandomHorizontalFlip(), ToTensor(), normalize]
)
_val_transforms = Compose(
    [Resize(size), CenterCrop(size), ToTensor(), normalize]
)

def train_transforms(examples):
    examples['pixel_values'] = [_train_transforms(image.convert("RGB")) for image in examples['image']]
    return examples

def val_transforms(examples):
    examples['pixel_values'] = [_val_transforms(image.convert("RGB")) for image in examples['image']]
    return examples

train_ds.set_transform(train_transforms)
val_ds.set_transform(val_transforms)

def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example["label"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}

model = ViTForImageClassification.from_pretrained(
    model_name_or_path,
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id
)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return dict(accuracy=accuracy_score(predictions, labels))

args = TrainingArguments(
    f"custom-vit-finetune",
    save_strategy="epoch",
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=10,
    per_device_eval_batch_size=4,
    num_train_epochs=5,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    logging_dir='logs_custom',
    remove_unused_columns=False,
)

trainer = Trainer(
    model,
    args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    data_collator=collate_fn,
    compute_metrics=compute_metrics,
    tokenizer=processor,
)

print("\n--- 커스텀 데이터셋 파인 튜닝 시작 ---")
trainer.train()

print("\n--- 최종 모델 성능 평가 시작 ---")

results = trainer.evaluate(eval_dataset=val_ds)

print(results)

final_accuracy = results['eval_accuracy']
print(f"\n 최종 모델의 검증 정확도 (Accuracy): {final_accuracy:.4f}")

# 학습 완료 후 모델 저장 (선택 사항)
output_dir = "final_custom_vit_model"
trainer.model.save_pretrained(output_dir)
processor.save_pretrained(output_dir)
print(f"\n 최종 모델이 {output_dir}에 저장되었습니다.")


# =======================================================
# 🌟 6. Post-Training Quantization (PTQ) 코드 수정 (에러 해결)
# =======================================================

QUANTIZED_OUTPUT_DIR = "final_quantized_vit_model_int8"
# 양자화된 모델의 가중치를 저장할 파일 이름 (Hugging Face의 기본 이름이 아님)
QUANTIZED_WEIGHTS_FILENAME = "quantized_pytorch_model.pt" 

print(f"\n--- Post-Training Quantization (INT8) 시작 ---")

try:
    # 1. 학습 완료된 모델을 불러옵니다.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 모델을 저장된 폴더에서 다시 로드 (ViTForImageClassification 객체)
    model_to_quantize = ViTForImageClassification.from_pretrained(output_dir)
    model_to_quantize.to(device)
    model_to_quantize.eval()

    # 2. 모델을 8비트 정수(INT8)로 양자화합니다.
    quantized_model = torch.quantization.quantize_dynamic(
        model_to_quantize, 
        {torch.nn.Linear, torch.nn.Conv2d}, 
        dtype=torch.qint8
    )

    print("✅ 모델 동적 양자화(Dynamic Quantization) 완료.")

    # 3. 양자화된 모델 저장 (PyTorch 표준 방식으로 저장)
    if not os.path.exists(QUANTIZED_OUTPUT_DIR):
        os.makedirs(QUANTIZED_OUTPUT_DIR)
        
    # a. config 파일과 processor 파일은 transformers 방식으로 저장
    model_to_quantize.config.save_pretrained(QUANTIZED_OUTPUT_DIR)
    processor.save_pretrained(QUANTIZED_OUTPUT_DIR)
    
    # b. 양자화된 모델의 state_dict만 저장
    output_path = os.path.join(QUANTIZED_OUTPUT_DIR, QUANTIZED_WEIGHTS_FILENAME)
    torch.save(quantized_model.state_dict(), output_path)

    print(f"✅ 양자화된 모델 가중치(state_dict)가 {output_path}에 저장되었습니다. (INT8)")
    
    print("\n💡 로드 시에는 일반 모델을 로드한 후, 저장된 state_dict를 로드하고 다시 양자화를 적용해야 합니다.")

except Exception as e:
    print(f"❌ 양자화 과정 중 오류 발생: {e}")