# pytorch 공부

---

## 개념 정리

### 텐서(tensor)

**텐서 = 다차원 배열 (multi-dimensional array)**

PyTorch의 torch.Tensor는 GPU에서 연산할 수 있도록 설계된 넘파이 배열의 확장판

- GPU에서 연산 가능 (.to('cuda'))

- 자동 미분 가능 (requires_grad=True)

- 딥러닝 모델에 바로 넣기 최적화

---

### batch

batch = 텐서 묶음 단위 = 한 번에 모델에 집어넣어 학습시키는 데이터 묶음

PyTorch에서는 배치 차원이 텐서의 첫 번째 차원으로 들어감

- 예시:
  - 흑백 이미지 한 장 → [1, 28, 28] (C,H,W)
  - 이미지 32장 묶음(batch) → [32, 1, 28, 28] (N,C,H,W)


#### 차원 표기 (이미지의 경우)

PyTorch: (N, C, H, W)

TensorFlow/Keras: (N, H, W, C)

-> 데이터셋을 가져올 때 프레임워크 맞게 차원을 바꿔줘야 할 때도 있음

- **N (Batch size)** : 한 번에 학습하는 데이터 묶음의 개수

- C (Channels): 이미지의 색상 채널 수 (예: 흑백=1, RGB=3)

- H (Height): 세로 방향 픽셀 수

- W (Width): 가로 방향 픽셀 수


---------

### 컨테이너(Container)

여러 레이어를 묶어 관리하는 클래스

- nn.Sequential: 순차적으로 레이어를 쌓음 (forward 정의 필요 없음/forward 과정을 자동으로 처리)
- nn.ModuleList: 리스트처럼 레이어를 보관. 반복문으로 접근 가능.
- nn.ModuleDict: 딕셔너리처럼 이름 붙여 관리.

---

### forward

nn.Module을 상속받아 모델을 만들 때, 입력 → 출력 계산 과정을 정의하는 함수.

즉, **순전파(forward propagation)** 를 어떻게 할지 정하는 부분.

model(x)를 호출하면 내부적으로 forward(x)가 실행.

>forward가 아래 흐름을 코드로 어떻게 계산할지 정함
>>  1. 입력 데이터 x가 들어옴
>> 
>>  2. 선형 변환(가중치·편향)
>> 
>>  3. 활성화 함수(ReLU, Sigmoid 등)
>> 
>>  4. 다음 레이어로 전달
>> 
>>  5. 마지막에 예측 결과 출력

**역전파(backward)** 는 PyTorch가 `autograd`로 자동 계산

사용자가 직접 오버라이드(override)해서 작성
- forward 안에서:
  - 레이어 호출 (self.fc1(x))
  - 활성화 함수 (F.relu)
  - 텐서 연산 (x.view)

-----

## 모듈 정리

### [**torch.utils.data**](https://docs.pytorch.org/docs/stable/data.html)

  데이터셋과 배치를 다루는 기본 모듈.

  Dataset 클래스를 기반으로 커스텀 데이터셋을 정의하고, DataLoader를 통해 배치 단위로 불러오며, Sampler를 통해 데이터 샘플링 방식을 제어
  
- 주요 클래스 & 함수
  - **`Dataset`**: 모든 데이터셋의 기본 클래스 (상속 받음)
    -  `__len__`(self): 전체 데이터 개수 반환
    - `__getitem__`(self, idx): 인덱스에 해당하는 데이터 반환 
  - **`DataLoader`**: 데이터를 배치 단위로 불러옴
    - DataLoader는 이 Dataset을 감싸서 배치(batch) 단위로 구성하고 반복(iterable) 가능한 객체로 만듬
  - **`TensorDataset`**: 여러 텐서를 묶어 하나의 데이터셋으로 사용
  - **`Sampler`** 계열: 데이터 샘플링 방식 제어
  - **`Subset`**: 일부 인덱스만 선택
  - **`ConcatDataset`**: 여러 데이터셋 합치기
  - **`random_split`**: 데이터셋을 랜덤하게 분할
  - **`IterableDataset`**: 크기 고정이 없는 스트리밍 데이터용

  Dataset → 데이터 정의
  DataLoader → 배치 단위 로딩
  Sampler → 순서/샘플링 방식 제어
  Subset/Concat/random_split → 데이터셋 조작
  
```python
DataLoader(dataset, batch_size=1, shuffle=False, sampler=None,
           batch_sampler=None, num_workers=0, collate_fn=None,
           pin_memory=False, drop_last=False, timeout=0,
           worker_init_fn=None, *, prefetch_factor=2,
           persistent_workers=False)

# 주요 파라미터
# dataset: 불러올 데이터셋 객체
# batch_size: 배치 크기 (기본값=1)
# shuffle: 데이터 순서를 섞을지 여부
# num_workers: 데이터를 병렬로 불러올 프로세스 개수
# drop_last: 마지막 불완전 배치를 버릴지 여부
# collate_fn: 배치 구성 방식을 지정하는 함수
```

---

### [torch.nn](https://docs.pytorch.org/docs/stable/nn.html)

**PyTorch의 신경망(neural network) 관련 모듈**

신경망을 쉽게 만들 수 있도록 **레이어, 손실 함수, 활성화 함수, 컨테이너(Sequential 등)** 제공

- 주요 구성 요소
  - 모듈 기반 (nn.Module)
    - 모든 신경망의 기본 클래스, 모델을 만들 때 상속받아 사용
  - 레이어 (Layers)
    - 선형(fully connected): nn.Linear
    - 합성곱(Conv): nn.Conv2d, nn.Conv1d, nn.Conv3d
    - 순환(RNN): nn.RNN, nn.LSTM, nn.GRU
    - 정규화: nn.BatchNorm2d, nn.LayerNorm, nn.Dropout
    - FFN은 2개의 선형 레이어 + 활성화 함수로 구성
    - MoE는 nn.ModuleList + nn.Linear + 샘플링 로직을 조합하면 직접 구현 가능
  - 활성화 함수
    - nn.ReLU, nn.Sigmoid, nn.Tanh, nn.Softmax 등
  - 손실 함수 (Loss Functions)
    - 회귀: nn.MSELoss
    - 분류: nn.CrossEntropyLoss, nn.NLLLoss
    - 그 외: nn.BCELoss, nn.HingeEmbeddingLoss 등
  - 컨테이너 (Containers)
    - nn.Sequential: 여러 레이어를 순서대로 쌓음
    - nn.ModuleList: 파이썬 리스트처럼 모듈 관리
    - nn.ModuleDict: 딕셔너리처럼 모듈 관리
    
---

### [**torch.nn.Module**](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html)

PyTorch 모든 신경망 레이어와 모델의 부모 클래스

신경망을 만들 때 **“구성(init) + 계산 과정(forward)”**을 정의하는 기본 틀

- 핵심 메서드
  - `__init__`(self)
    - 레이어, 가중치 같은 구성 요소 정의.
  - `forward(self, x)`
    - 입력 → 출력 변환 과정을 정의 / 실제 학습/추론 시 호출되는 함수.
    - `model(x)` 하면 내부적으로 `forward()`가 실행.
  - `parameters()`
    - 모델 안에 있는 학습 가능한 파라미터(가중치, 편향 등) 반환.
    - 옵티마이저에 넣어서 학습할 때 사용.
  - `to(device)`
    - 모델 전체를 지정한 장치(CPU, GPU)로 옮김.
   
#### 기본 구조 예시
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(28*28, 128)  # 입력층
        self.fc2 = nn.Linear(128, 10)     # 출력층

    def forward(self, x):
        x = x.view(-1, 28*28)   # 2D 이미지를 1D로 펼치기
        x = F.relu(self.fc1(x)) # 은닉층 + 활성화
        x = self.fc2(x)         # 출력층
        return x

# 모델 생성
model = SimpleNet()
print(model)

# 입력 데이터
x = torch.randn(64, 1, 28, 28)  # 배치 64, 흑백 이미지
out = model(x)
print(out.shape)  # torch.Size([64, 10])

```

-----
