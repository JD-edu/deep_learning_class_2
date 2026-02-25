# datasets isbelong to huggingface. 
from datasets import load_dataset
import cv2
import numpy as np

# Full CIFAR10 is almost 70000 (60000: train 10000:test)
# We download it partially. 
train_ds, test_ds = load_dataset('cifar10', split=['train[:1000]', 'test[:200]'])
# to check shape of data 
#print(train_ds.shape)
#print(test_ds.shape)
#print(type(train_ds))

# Split up training into training + validation 
splits = train_ds.train_test_split(test_size=0.1)
#print(splits)
train_ds = splits['train']
val_ds = splits['test']
# check train_ds 
#print(train_ds)
#print(train_ds.features)

#print(train_ds[231]['img'])
#print(train_ds[231]['label'])

# Check image visually - Using OpenCV
pil_img = train_ds[231]['img']
img_np = np.array(pil_img)
img_bgr  = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
cv2.imshow('win', img_bgr)
cv2.waitKey(2000)
cv2.destroyAllWindows()#

# Using Matplotlib
#import matplotlib.pyplot as plt 
#plt.imshow(img_np)
#plt.title(f"Image 231 - Label: {train_ds[231]['label']}")
#plt.axis('off')
#plt.show()

# find label actual names
id2label = {id:label for id, label in enumerate(train_ds.features['label'].names)}
label2id = {label:id for id,label in id2label.items()}
#print(train_ds.features)
#id2label = {}
#for id, label in enumerate(train_ds.features['label'].names):
#    id2label[id] = label
#label2id = {label:id for id,label in id2label.items()}

print(id2label)
print(label2id)

print(id2label[train_ds[231]['label']])

from transformers import ViTImageProcessor 

processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")

from torchvision.transforms import (CenterCrop,
                                    Compose,
                                    Normalize,
                                    RandomHorizontalFlip,
                                    RandomResizedCrop,
                                    Resize,
                                    ToTensor)

image_mean, image_std = processor.image_mean, processor.image_std
size = processor.size["height"]

normalize = Normalize(mean=image_mean, std=image_std)
# 트레인 데이터의 가상화와 검증 데이터의 가상화가 약간 다름 
_train_transforms = Compose(
        [
            RandomResizedCrop(size),
            RandomHorizontalFlip(),
            ToTensor(),
            normalize,
        ]
    )

_val_transforms = Compose(
        [
            Resize(size),
            CenterCrop(size),
            ToTensor(),
            normalize,
        ]
    )

def train_transforms(examples):
    examples['pixel_values'] = [_train_transforms(image.convert("RGB")) for image in examples['img']]
    return examples

def val_transforms(examples):
    examples['pixel_values'] = [_val_transforms(image.convert("RGB")) for image in examples['img']]
    return examples

from torch.utils.data import DataLoader
import torch

def collate_fn(examples):
    # 전체 입력 이미지를 적층하는 역할 
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    # 라벨을 텐서화 (label binizer와 비슷)
    labels = torch.tensor([example["label"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}

# Set the transforms
train_ds.set_transform(train_transforms)
val_ds.set_transform(val_transforms)
test_ds.set_transform(val_transforms)

from transformers import ViTForImageClassification

model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224-in21k',
                                                  id2label=id2label,
                                                  label2id=label2id)

from transformers import TrainingArguments, Trainer

metric_name = "accuracy"

args = TrainingArguments(
    f"test-cifar-10",
    save_strategy="epoch",
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=1,
    num_train_epochs=3,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model=metric_name,
    logging_dir='logs',
    remove_unused_columns=False,
)


from sklearn.metrics import accuracy_score
import numpy as np

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return dict(accuracy=accuracy_score(predictions, labels))

import torch

trainer = Trainer(
    model,
    args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    data_collator=collate_fn,
    compute_metrics=compute_metrics,
    tokenizer=processor,
)

trainer.train()

"""## Evaluation

Finally, let's evaluate the model on the test set:
"""

outputs = trainer.predict(test_ds)

print("트레인한 결과 : ", outputs.metrics)

"""We can also easily create a confusion matrix:"""

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

y_true = outputs.label_ids
y_pred = outputs.predictions.argmax(1)

labels = train_ds.features['label'].names
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot(xticks_rotation=45)

# 1. 저장할 폴더 이름 설정
output_dir = "./"

# 2. 학습이 완료된 최종 모델을 지정된 폴더에 저장
# trainer.model은 학습된 가중치와 구조를 포함합니다.
print(f"Saving final model to {output_dir}...")
trainer.model.save_pretrained(output_dir)

# 3. 모델과 일관성을 유지하도록 전처리 설정(processor)도 함께 저장
# 재사용 시 모델이 요구하는 정확한 이미지 전처리(크기, 정규화)를 보장합니다.
processor.save_pretrained(output_dir)

print("Model and processor saved successfully.")
