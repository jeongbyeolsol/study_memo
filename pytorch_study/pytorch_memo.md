# pytorch 공부

---

## PyTorch 학습 파이프라인 흐름
1. **모델 구조 정의**
  - `__init__`: 레이어(파라미터 포함) 선언
  - `forward`: 데이터가 흘러가는 연산 정의

2. **손실 함수(loss function) & 최적화 알고리즘(optimizer) 정의**
  - 손실 함수: 예측값과 정답 비교 (`nn.CrossEntropyLoss`, `nn.MSELoss` 등)
  - 옵티마이저: 파라미터 업데이트 규칙 정의 (`optim.SGD`, `optim.Adam` 등)

3. **데이터 준비**
  - Dataset + DataLoader
  - CPU/GPU 어디서 연산할지 device 결정 → .to(device)로 옮김

4. **순전파 (forward pass)**
  - `pred = model(X)`
  - 입력 → 레이어 통과 → 출력(logits)

5. **손실 계산**
  - `loss = loss_fn(pred, y)`
  - 모델 출력과 정답 비교 → 스칼라 손실값

6. **역전파 (backward pass)**
  - `loss.backward()`
  - 오차를 바탕으로 각 파라미터의 gradient 계산

7. **파라미터 갱신 (update)**
  - `optimizer.step()` → gradient 기반으로 weight/bias 업데이트
  - `optimizer.zero_grad()` → gradient 초기화 (누적 방지)

8. **반복 학습**
  - 위 과정을 여러 배치, 여러 epoch 동안 반복
  - 손실이 줄고, 모델 성능이 안정화될 때까지 학습

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

한 번의 forward/backward pass에서 처리하는 샘플(데이터) 개수

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

### 손실 함수 (Loss function)

모델 출력 y'와 정답 y 사이의 차이를 수치로 계산.


### 옵티마이저 (Optimizer)

모델 파라미터(가중치, 편향)를 어떻게 업데이트할지 정의하는 알고리즘.



### 순전파 (Forward propagation)

이터를 모델에 넣었을 때, 입력 → 출력으로 신호가 흘러가는 과정.

1. 입력 x (이미지, 문장 등)이
2. 가중치 W, 편향 b를 거치고
3. 활성화 함수 σ를 통과해
4. 최종 출력 y'가 나옴.

`pred = model(X)` = 순전파


### 역전파 (Backward propagation)

손실을 줄이려면 모델 파라미터(가중치, 편향)를 조정해야 함.

어떤 방향으로 얼마나 바꿀지 알려주는 게 미분(gradient).

1. **연쇄 법칙(chain rule)**\으로 손실을 파라미터까지 미분
2. 각 파라미터에 대한 기울기(gradient = ∂L/∂W, ∂L/∂b)를 계산
3. PyTorch는 이 과정을 loss.backward()에서 자동으로 처리
4. Optimizer가 gradient를 보고 파라미터를 수정.

----

`model.eval()`, `model.train()`에 영향받는 레이어

### Dropout

- 역할
  - **과적합 방지(regularization)**
  - 학습할 때 신경망의 일부 뉴런을 확률적으로 끄는(0으로 만드는) 기법.
- 학습 모드(`model.train()`)
  - 각 뉴런을 일정 확률(p)로 랜덤하게 꺼버림.
  - 남은 뉴런은 값이 커지지 않게 scale 보정.
- 추론 모드(`model.eval()`)
  - 뉴런을 끄지 않고 전부 활성화.
  - 학습 시의 scale 보정을 그대로 반영해서 일관된 출력을 냄.

### Batch Normalization (BatchNorm)

- 역할
  - 학습 중 각 배치의 평균과 분산을 계산해서 데이터를 정규화(normalization).
  - 학습 안정화, 수렴 속도 향상.
- 학습 모드(`model.train()`)
  - 현재 들어온 **배치(batch)**의 평균/분산을 사용.
  - 동시에 "러닝 평균/분산"을 업데이트해서 나중에 추론 때 쓸 준비.
- 추론 모드(`model.eval()`)
  - 배치에서 평균/분산을 새로 계산하지 않고, 학습 때 모아둔 러닝 평균/분산을 사용.

---

### Epoch

**1 Epoch** = 훈련 데이터 전체를 한 번 모델에 통과시켜 학습하는 것.

딥러닝에서는 보통 수십~수백 epoch 동안 데이터를 반복 학습시킴.

이유: 신경망은 파라미터가 많고 비선형 구조라, 데이터 한 번만으로는 충분히 학습되지 않음.

**Iteration**: 한 번의 미니배치(batch)를 모델에 통과시키고 가중치를 업데이트하는 과정.

#### Q: 적절한 Epoch은 어떻게 정하나?

1. **조기 종료(Early Stopping)**
  - 학습/검증 손실을 모니터링하다가 검증 손실이 더 이상 줄지 않고 오히려 늘어나기 시작할 때 멈춤.
  - 과적합을 막으면서 가장 좋은 지점에서 멈출 수 있음.

2. **학습 곡선(Learning Curve) 확인**
  - Epoch마다 훈련 손실, 검증 손실, 정확도를 그래프로 기록.
  - 검증 손실이 바닥을 찍고 다시 올라가기 전까지가 적절한 Epoch.

3. **경험적 설정**
  - 작은 모델/데이터 → 보통 10~50 Epoch.
  - 큰 모델/데이터 → 보통 100 Epoch 이상.
  - 최적 값은 데이터와 모델 복잡도에 따라 다름.

4. **하이퍼파라미터 튜닝**
  - Grid Search, Random Search, Bayesian Optimization 같은 방법으로 Epoch 수를 포함해 실험.

---

## 함수 정리

### `zero_grad()`

gradient 초기화

PyTorch의 기본 동작은 **gradient를 누적(accumulate)** 일반적인 학습 루프에서는 각 배치마다 gradient를 새로 계산해야 함.


### `model.eval()`

모델을 추론(inference) 모드로 바꿈

- 영향을 받는 레이어:

  - Dropout → 끔 (훈련 시엔 일부 뉴런을 랜덤하게 끄지만, 추론 시엔 전부 사용)

  - BatchNorm → 러닝 평균/분산 값 사용 (훈련 시엔 배치 통계 사용)

### `torch.no_grad():`

autograd(자동 미분 엔진)를 끔. (PyTorch는 기본적으로 모든 텐서 연산을 추적해서 연산 그래프(computation graph)를 만듬)

1. .backward()를 안 하니까 gradient 저장 안 함

2. 메모리 절약 (gradient 계산 그래프를 안 만듦)

3. 연산 속도 향상


### `detach()`

특정 텐서를 autograd 추적에서 분리 (gradient 계산 X).

### `torch.autograd.grad(outputs, inputs)`

특정 연산에 대한 gradient를 바로 리턴. (모델 파라미터 전체 말고 일부만 필요할 때)

### `torch.autograd.backward(tensors, grad_tensors=None)`

여러 텐서에 대해 한꺼번에 backward 실행.

---

모델을 정의할 때 필수로 만들 것

### `__init__`

Python 클래스의 생성자(constructor)

모델의 구성 요소(레이어, 파라미터 등)를 정의

-> forward에서 그 구성 요소와 연산을 어떻게 연결해서 데이터를 흘려보낼지 정의

### `forward`

nn.Module을 상속받아 모델을 만들 때, 입력 → 출력 계산 과정을 정의하는 함수.

즉, **순전파(forward propagation)**/를 어떻게 할지 정하는 부분.

model(x)를 호출하면 내부적으로 forward(x)가 실행. (`__call__` 메서드를 정의해두면, 그 객체를 “함수처럼” 호출할 수 있음)

**역전파(backward)** 는 PyTorch가 `autograd`로 자동 계산

>사용자가 직접 오버라이드(override)해서 작성
>>forward 안에서:
>>- 레이어 호출 (self.fc1(x))
>>- 활성화 함수 (F.relu)
>>- 텐서 연산 (x.view)

-----
   
### `.item()`

PyTorch 텐서를 **파이썬 숫자(float, int)**\로 변환하는 메서드.

조건: 텐서 안에 값이 정확히 1개일 때만 가능.

```python
x = torch.tensor([3.14])
print(x.item())       # 3.14 (float)

y = torch.tensor(7)
print(y.item())       # 7 (int)

```

### `.argmax(dim)`

argmax = "가장 큰 값의 인덱스"를 반환하는 함수

PyTorch에서는 `tensor.argmax(dim=n)`처럼 쓰고, 어떤 차원에서 찾을지를 dim으로 정함.

`x.argmax(dim=0)` → 열(column) 기준 / `x.argmax(dim=1)` → 행(row) 기준

pred는 `[batch, num_classes]` 점수 행렬 -> 가장 높은 점수를 받은 클래스가 예측

```python
pred = tensor([
  [0.1, 0.7, 0.2],   # 샘플 1에 대한 클래스 0,1,2 점수
  [0.9, 0.05, 0.05], # 샘플 2
  [0.2, 0.1, 0.7],   # 샘플 3
  [0.3, 0.3, 0.4]    # 샘플 4
])

pred.argmax(1)  # tensor([1, 0, 2, 2])
```

----

### `torch.save`

PyTorch 객체(텐서, 딕셔너리, 모델 가중치 등)를 파일로 직렬화해서 저장하는 함수.

형식은 .pth 또는 .pt 확장자를 주로 씀 (실제로는 단순한 바이너리 파일).

```python
torch.save(obj, "경로/이름.pth")
# obj: 저장할 객체 (텐서, 리스트, dict, state_dict 등)
# "파일명": 저장될 파일 이름
```

### `torch.load`

저장된 PyTorch 객체를 디스크에서 읽어오는 함수.

`torch.save`로 저장한 걸 다시 메모리로 불러올 때 사용.

```python
torch.load(f, map_location=None)
```


### `model.state_dict()`

**모델의 모든 학습 가능한 파라미터(가중치와 편향)와 버퍼(예: BatchNorm 통계)**\를 딕셔너리 형태로 반환.

키(key) = 레이어 이름 / 값(value) = 파라미터 텐서.

```python
torch.save(model.state_dict(), "model.pth")
```

모델 전체를 저장하면(PyTorch 객체 자체 저장) → 파이썬 코드 버전, 경로 등에 의존성이 생겨서 이식성이 떨어짐.

반면 `state_dict()`는 순수한 파라미터 값만 저장하니까, 나중에 같은 모델 클래스를 새로 정의한 후 `load_state_dict`로 불러오기만 하면 됨.

즉, “코드와 파라미터를 분리”할 수 있음.

### `load_state_dict`

`load_state_dict`는 `state_dict`를 다시 모델에 불러오는 함수

모델 인스턴스에 파라미터(state_dict)를 주입하는 메서드.

- 흐름
  
  - `model.state_dict()` → **모델의 학습된 파라미터(가중치/편향 등)**\를 딕셔너리로 반환

  - `torch.save(...)` → 그걸 파일로 저장

  - `model.load_state_dict(...)` → 저장된 파라미터 dict을 다시 모델에 불러오기 (인자로 **state_dict 객체(OrderedDict)**\를 받음)


```python
# 저장
torch.save(model.state_dict(), "model.pth")

# 불러오기
model = NeuralNetwork()                           # 같은 구조의 모델 인스턴스 필요
model.load_state_dict(torch.load("model.pth"))    # 가중치 로드
model.eval()                                      # 추론 모드 전환
```

---

### 형변환 메서드

| 방법                   | 설명                        | 특징              |
| -------------------- | ------------------------- | --------------- |
| `.type(torch.float)` | dtype을 명시적으로 바꿈           | 옛날 스타일, 가독성 ↓   |
| `.to(torch.float)`   | dtype 변경 + device 이동까지 가능 | 가장 범용적          |
| `.float()`           | float32로 변환하는 축약형         | 간단하지만 dtype 고정됨 |

---

## 클래스 정리

### `DataLoader` [`torch.utils.data.DataLoader`](https://docs.pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader)

**batch 단위로 데이터를 꺼낼 수 있는 반복자(iterator)**

torch.utils.data.DataLoader

```python
DataLoader(dataset, batch_size=1, shuffle=False, sampler=None,
           batch_sampler=None, num_workers=0, collate_fn=None,
           pin_memory=False, drop_last=False, timeout=0,
           worker_init_fn=None, *, prefetch_factor=2,
           persistent_workers=False)

# 주요 파라미터
# dataset: 불러올 데이터셋 객체
# batch_size: 배치 크기 (기본값=1)
# shuffle: 데이터 순서를 섞을지 여부 / 전체 데이터셋의 순서를 먼저 섞고(shuffle), 그걸 batch_size 단위로 잘라서 배치를 만듦.
# num_workers: 데이터를 병렬로 불러올 프로세스 개수
# drop_last: 마지막 불완전 배치를 버릴지 여부
# collate_fn: 배치 구성 방식을 지정하는 함수
```
  
---

### [`torch.utils.data.Dataset`](https://docs.pytorch.org/docs/stable/data.html#torch.utils.data.Dataset)

커스텀 데이터셋을 만들 때 상속하는 기본 

- 필수로 구현해야 할 메서드:
  - `__init__`: 초기화 (이건 꼭 필수는 아님)
  -  `__len__`(self): 전체 데이터 개수 반환
  - `__getitem__`(self, idx): 인덱스에 해당하는 데이터 반환




----

### [`torch.nn.Module`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html)

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
----

### `torch.device`

텐서나 모델이 어느 장치(CPU, GPU, MPS 등) 위에 있는지를 나타내는 객체

```python
# 생성 방법
import torch

cpu_device = torch.device("cpu")        # CPU
cuda_device = torch.device("cuda")      # GPU (기본은 0번)
cuda1_device = torch.device("cuda:1")   # 두 번째 GPU
mps_device = torch.device("mps")        # Apple Silicon (Mac)
```


```python
# 사용 예시
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 텐서를 해당 장치로 이동
x = torch.randn(3, 3) 
x = x.to(device)      # X의 값을 GPU 메모리로 복사해서, device 속성이 cuda/cpu인 새 텐서를 반환

# 모델도 이동
model = torch.nn.Linear(3, 3)
model = model.to(device)

print(x.device)     # cuda:0 또는 cpu
print(next(model.parameters()).device)  # 모델 파라미터가 위치한 장치

```
-----

## 모듈 정리

### [`torch.utils.data`](https://docs.pytorch.org/docs/stable/data.html)

  데이터셋과 배치를 다루는 기본 모듈.

  Dataset 클래스를 기반으로 커스텀 데이터셋을 정의하고, DataLoader를 통해 배치 단위로 불러오며, Sampler를 통해 데이터 샘플링 방식을 제어
  
- 주요 클래스 & 함수
  - **`Dataset`**: 모든 데이터셋의 기본 클래스 (상속 받음) 
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

---


### [`torch.nn`](https://docs.pytorch.org/docs/stable/nn.html)

**PyTorch의 신경망(neural network) 관련 모듈**

신경망을 쉽게 만들 수 있도록 **레이어, 손실 함수, 활성화 함수, 컨테이너(Sequential 등)** 제공

- 주요 구성 요소
  - 모듈 기반 (nn.Module)
    - 모든 신경망의 기본 클래스, 모델을 만들 때 상속받아 사용
  - 레이어 (Layers)
    - 선형(fully connected): `nn.Linear`
    - 합성곱(Conv): `nn.Conv2d`, `nn.Conv1d`, `nn.Conv3d`
    - 순환(RNN): `nn.RNN`, `nn.LSTM`, `nn.GRU`
    - 정규화: `nn.BatchNorm2d`, `nn.LayerNorm`, `nn.Dropout`
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

```python
  class torch.nn.Flatten(start_dim=1, end_dim=-1)
# start_dim (기본=1): 어디서부터 flatten할지 시작 차원
# end_dim (기본=-1): 어디까지 flatten할지 끝 차원
# start_dim~end_dim 사이의 모든 차원을 1D로 펴버림.
```

---


### [`torch.optim`](https://docs.pytorch.org/docs/stable/optim.html)

PyTorch에서 신경망 학습 시 파라미터를 업데이트하는 알고리즘들을 모아둔 모듈.

예: SGD, Adam, RMSprop 등.

모델의 파라미터(model.parameters())를 받아서 gradient 기반으로 업데이트

---

### `torch.autograd`

PyTorch의 **자동 미분(automatic differentiation) 엔진**을 모아둔 모듈.

텐서 연산을 추적해서 **연산 그래프(computational graph)**\를 만들고, `.backward()` 호출 시 그래프를 따라 미분값(gradient)을 계산해서 저장.

이 gradient를 옵티마이저(`torch.optim`)가 사용해서 파라미터를 업데이트

- 텐서 생성 시 **`requires_grad=True`**로 설정하면 autograd가 이 텐서의 연산 과정을 추적.

- gradient 결과는 **`.grad`** 속성에 누적(accumulate)됨.

-  `y.backward()`
 ```python
x = torch.tensor([2.0], requires_grad=True)
y = x ** 2
y.backward()
print(x.grad)  # tensor([4.])
 ```

---
