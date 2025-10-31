# torch call graph

![call_graph](,/images/torch_callgraph.png)

### 거시적인 흐름의 1 step call graph
```
DataLoader → (x, y)
      │
      ▼
nn.Module (model) ── forward(x) ──► logits
      │                              │
      │                           LossFn(logits, y) → loss (scalar)
      │                              │
      └─────────── Autograd: loss.backward() ◄──────┘
                        │
     각 연산의 grad 계산(그래프 역추적) → Parameter.grad에 누적
                        │
                Optimizer.step()
                        │
              (옵션) LR Scheduler.step()
                        │
              Optimizer.zero_grad()

```

세로선 포함 직사각형: 모듈

사각형: class name

삼각형: 임의의 클래스

동그라미: 함수, 메서드


---

### trace.json파일 보는법

trace.json파일은 draw_callback.ipynb에서 매우 간단한 모델을 실행하면서 torch.profiler(모델의 연산 단위별 실행 시간, 메모리 사용량, 커널 호출 등을 기록해주는 전역 분석기 모듈)을 통해 얻은 성능 그래프 파일임

torch.profiler는 PyTorch 연산의 실행 시간·메모리·커널을 추적해주는 성능 분석용 모듈이며, 내부적으로 전역 프로파일링 엔진을 제공한다

1. tracing파일을 다운받기
2. chrome://tracing/을 크롬 주소창에 검색 (권한 허용이 필요할 수 있음)
3. 좌측 상단에 Load를 눌러 tracing파일을 선택

아래 이미지는 tracing파일을 간략하게 정리한것

![simple_example_call_graph](,/images/simple_example_call_graph.png)
