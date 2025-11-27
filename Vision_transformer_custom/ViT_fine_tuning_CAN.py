# -*- coding: utf-8 -*-

import torch
import numpy as np
from PIL import Image
from transformers import ViTForImageClassification, ViTImageProcessor, TrainingArguments, Trainer
from torchvision.transforms import (Compose, Normalize, RandomHorizontalFlip, RandomResizedCrop, ToTensor, Resize, CenterCrop)
from torch.utils.data import DataLoader


CUSTOM_DATA_ROOT = "./images"  # 'can', 'pet', 'cup' 폴더가 포함된 상위 폴더

# Hugging Face datasets의 'imagefolder' 기능을 사용하여 폴더 구조를 바로 로드
# train/val/test 폴더를 명시적으로 분리하지 않았다면, 하나의 train split으로 로드됩니다.
# 이 경우, 이후 단계에서 train_test_split으로 분리해야 합니다.
from datasets import load_dataset
# 로컬 폴더를 로드할 때는 'imagefolder'와 data_dir 인수를 사용합니다.
raw_datasets = load_dataset('imagefolder', data_dir=CUSTOM_DATA_ROOT)

# 2. 훈련/검증 데이터 분리 (필수)
# 로드된 전체 데이터셋('train' split)을 90%는 훈련, 10%는 검증으로 분리
splits = raw_datasets['train'].train_test_split(test_size=0.1, seed=42)
train_ds = splits['train']
val_ds = splits['test']

# 3. 클래스 정의 및 매핑 (수정된 부분)
# ImageFolder 로더는 폴더 이름을 기반으로 자동으로 클래스를 정의합니다.
labels = train_ds.features['label'].names
id2label = {i: label for i, label in enumerate(labels)}
label2id = {label: i for i, label in enumerate(labels)}
num_labels = len(labels) # 3 (can, pet, cup)

print(f"✅ 로드된 클래스: {labels}")
print(f"✅ 총 훈련 데이터 개수: {len(train_ds)}")

model_name_or_path = 'google/vit-base-patch16-224-in21k'
processor = ViTImageProcessor.from_pretrained(model_name_or_path)

image_mean, image_std = processor.image_mean, processor.image_std
size = processor.size["height"]

# 전처리 파이프라인 정의 (CIFAR-10 코드와 동일)
normalize = Normalize(mean=image_mean, std=image_std)
_train_transforms = Compose(
    [RandomResizedCrop(size), RandomHorizontalFlip(), ToTensor(), normalize]
)
_val_transforms = Compose(
    [Resize(size), CenterCrop(size), ToTensor(), normalize]
)

def train_transforms(examples):
    # 'image' 컬럼은 ImageFolder의 기본 출력 컬럼입니다.
    examples['pixel_values'] = [_train_transforms(image.convert("RGB")) for image in examples['image']]
    return examples

def val_transforms(examples):
    examples['pixel_values'] = [_val_transforms(image.convert("RGB")) for image in examples['image']]
    return examples

# 전처리 적용
train_ds.set_transform(train_transforms)
val_ds.set_transform(val_transforms)

def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example["label"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}

model = ViTForImageClassification.from_pretrained(
    model_name_or_path,
    num_labels=num_labels, # 3으로 자동 설정됨
    id2label=id2label,
    label2id=label2id
)

# =======================================================
# 📌 5. Trainer 설정 및 학습 실행 (기존 코드와 동일)
# =======================================================
from transformers import TrainingArguments, Trainer
from sklearn.metrics import accuracy_score

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return dict(accuracy=accuracy_score(predictions, labels))

args = TrainingArguments(
    f"custom-vit-finetune",
    save_strategy="epoch",
    eval_strategy="epoch", # 'evaluation_strategy' 대신 'eval_strategy' 사용
    learning_rate=2e-5,
    per_device_train_batch_size=10,
    per_device_eval_batch_size=4,
    num_train_epochs=5, # CIFAR-10보다 데이터가 적으므로 에포크를 늘릴 수 있음
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

# 1. 검증 데이터셋 (val_ds)에 대한 평가
# trainer.evaluate()를 호출하여 compute_metrics에서 정의한 정확도를 계산합니다.
results = trainer.evaluate(eval_dataset=val_ds)

# 2. 결과 출력
# results 딕셔너리에는 eval_loss, eval_accuracy 등의 정보가 포함되어 있습니다.
print(results)

# 3. 최종 정확도 명시적 출력
final_accuracy = results['eval_accuracy']
print(f"\n✅ 최종 모델의 검증 정확도 (Accuracy): {final_accuracy:.4f}")

# 학습 완료 후 모델 저장 (선택 사항)
output_dir = "final_custom_vit_model"
trainer.model.save_pretrained(output_dir)
processor.save_pretrained(output_dir)
print(f"\n✅ 최종 모델이 {output_dir}에 저장되었습니다.")