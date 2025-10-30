1. tracing파일을 다운받기
2. chrome://tracing/을 크롬 주소창에 검색 (권한 허용이 필요할 수 있음)
3. 좌측 상단에 Load를 눌러 tracing파일을 선택



거시적인 흐름의 1 step call graph
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
