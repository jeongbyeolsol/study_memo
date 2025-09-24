# torchvision

PyTorch의 공식 **컴퓨터 비전 라이브러리**.  
이미지/비디오 딥러닝 실험에 필요한 **데이터셋, 전처리, 모델, 입출력 기능**을 제공한다.

---

## 1. torchvision.datasets
유명한 벤치마크 데이터셋을 쉽게 불러오기 가능. (`download=True`로 자동 다운로드)

```python
from torchvision import datasets

# CIFAR-10 데이터셋
trainset = datasets.CIFAR10(
    root='./data',
    train=True,
    download=True
)
```

**대표적인 데이터셋**
- MNIST, FashionMNIST
- CIFAR10, CIFAR100
- ImageNet
- COCO, VOC

---

## 2. torchvision.transforms
이미지 전처리 및 데이터 증강 기능.

```python
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((224, 224)),          # 크기 조정
    transforms.RandomHorizontalFlip(),      # 좌우 반전
    transforms.ToTensor(),                  # Tensor 변환
    transforms.Normalize(mean=[0.5], std=[0.5])  # 정규화
])
```

**주요 transform**
- `Resize`, `CenterCrop`, `RandomCrop`
- `RandomHorizontalFlip`, `RandomRotation`
- `ColorJitter`
- `ToTensor`, `Normalize`

---

## 3. torchvision.models
사전 학습(pretrained)된 유명 모델 제공.

```python
from torchvision import models

# ResNet-18 (pretrained)
model = models.resnet18(pretrained=True)
```

**지원 모델**
- CNN 계열: AlexNet, VGG, ResNet, DenseNet, MobileNet
- Vision Transformer, Swin Transformer
- Detection/Segmentation: Faster R-CNN, Mask R-CNN, DeepLabV3 등

---

## 4. torchvision.io
이미지/영상 입출력 기능.

```python
from torchvision.io import read_image

img = read_image("example.jpg")   # [C,H,W] Tensor 반환
```

---

## 📌 정리
- **datasets** → 유명 데이터셋 로딩  
- **transforms** → 전처리 & 데이터 증강  
- **models** → pretrained 모델 사용  
- **io** → 이미지/영상 입출력  

---

👉 `torchvision`은 **데이터 → 전처리 → 모델 → 입출력**을 한 번에 다루게 해주는 패키지.  
따라서 **연구용 실험, Kaggle 대회, 논문 구현** 등에서 거의 필수적으로 사용된다.
