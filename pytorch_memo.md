# pytorch 공부

## 모듈 정리

- **torch.utils.data**

  데이터셋과 배치를 다루는 기본 모듈.

  Dataset 클래스를 기반으로 커스텀 데이터셋을 정의하고, DataLoader를 통해 배치 단위로 불러오며, Sampler를 통해 데이터 샘플링 방식을 제어

  - **`Dataset`** 모든 데이터셋은 이를 상속받아 구현
    -  `__len__`(self): 전체 데이터 개수 반환
    - `__getitem__`(self, idx): 인덱스에 해당하는 데이터 반환 
  - **`DataLoader`** 데이터셋을 배치 단위로 반복(iteration)할 수 있도록 해주고 병렬 처리, 셔플링, 배치 구성 등 다양한 옵션 지원.
  - 등등
  
```python
DataLoader(dataset, batch_size=1, shuffle=False, sampler=None,
           batch_sampler=None, num_workers=0, collate_fn=None,
           pin_memory=False, drop_last=False, timeout=0,
           worker_init_fn=None, *, prefetch_factor=2,
           persistent_workers=False)
#주요 파라미터
#dataset: 불러올 데이터셋 객체
#batch_size: 배치 크기 (기본값=1)
#shuffle: 데이터 순서를 섞을지 여부
#num_workers: 데이터를 병렬로 불러올 프로세스 개수
#drop_last: 마지막 불완전 배치를 버릴지 여부-
#collate_fn: 배치 구성 방식을 지정하는 함수
```
