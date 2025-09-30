![인코더 흐름](https://jalammar.github.io/images/t/transformer_multi-headed_self-attention-recap.png)

# 인코더 구조

인코더를 직렬로 쌓은 구조 (원 논문 6개)

인코더 내부에 **self-attention**, **FFN**이 존재

👉 멀티헤드 어텐션의 head들은 병렬이지만,

👉 인코더 블록 자체는 직렬로 연결돼서 위로 쌓임.

인코더 블록의 최종 출력은 항상 입력과 동일한 차원

# 디코더 구조

 **masked self-attention** -> **encoder-decoder-attention** -> **FFN**

- masked self-attention
  - 마스크 덕분에 각 토큰은 자신과 그 이전 토큰까지만 볼 수 있음

- **encoder-decoder-attention**
  - Query는 디코더의 은닉 상태, Key/Value는 인코더의 출력
 
디코더도 인코더와 마찬가지로 **출력 차원은 입력과 동일한 d_model**을 유지

다만 마지막 단계에서만 선형 변환으로 단어 분포 V차원으로 바꿈

# Attention

`attention(Q, K, V) = softmax(QK^T / d  + mask) * V`

X * (WQ, WK, WV) = (Q, K, V)  (각 인코더 블록마다 따로 존재하는 파라미터)

Q: Query vector

K: Key vector

V: Value vector

인코더/디코더 블록에 들어오는 토큰 벡터는 차원 d_model

멀티헤드 어텐션에서는 이걸 여러 “head”로 쪼갬.

h: head 개수
d_k = the dimension of the key vectors

`d_k = d_v = d_model / h`

## Self-Attention

각 단어 벡터를 다른 단어와 비교해서 “누구에게 집중할지” 가중치를 주는 메커니즘

## Multi-Head Attention

입력 벡터의 차원을 여러 head로 나눈

각 head는 자기만의 가중치(W_Q, W_L, W_V)를 써서 **Q**, **K**, **V**를 계산하고 어텐션을 구함.

-> head마다 서로 다른 표현 공간에서 문맥을 해석

모든 head의 출력을 합치고(concat), 다시 선형변환해서 최종 출력으로 만듦.

MultiHead(Q, K, V) = Concat(head_1, ..., head_h) * W_o

### head

멀티헤드 어텐션 안에 들어 있는 하나의 독립적인 어텐션 연산 단위.

멀티헤드에서의 동작

1. 입력(토큰 하나) 차원 d_model을 h개의 head로 쪼갬.
2. 각 head는 d_k = d_v = d_model / h을 차원에서 어텐션을 계산 (d_q == d_k)
3. Head별 출력들을 concat
4. 마지막에 W_O로 projection. (H * W_O) / 출력을 원래 차원으로 돌려놓는 역할 R^(h*d_v x d_model)

# 임베딩

텍스트(단어)를 숫자 벡터로 바꾸는 단계

1. 토큰화(tokenization): 문장을 단어/서브워드/문자 단위로 쪼갬.
2. 정수 인덱스 부여: 어휘(vocabulary) 딕셔너리를 만들어 각 토큰에 id를 붙임.
3. 임베딩 매트릭스 조회: 각 단어 id를 길이 d_model짜리 실수 벡터로 매핑

`embedding(x) = W * onehot(x)`
  
원-핫 벡터에서 1이 있는 위치를 뽑으면, 사실상 **임베딩 매트릭스의 해당 행(row)**\을 꺼내는 것과 동일

### W 학습의 두가지 방식

1. 트랜스포머 안에서 같이 학습하는 경우 (end-to-end)
    - 가장 기본적이고 흔한 방식. (현대에도 주로 사용)
    - 학습 과정에서 임베딩 벡터, 어텐션 가중치, FFN 파라미터 등이 같은 loss를 통해 동시에 업데이트됨.
   
2. 사전학습된 임베딩을 불러와 쓰는 경우 (pre-trained embeddings)
   - 과거에는 Word2Vec, GloVe, fastText처럼 트랜스포머와 별개로 학습된 임베딩을 불러와서 초기화하는 방법을 많이 사용함
   - 학습 초반에 의미 공간이 이미 잘 잡혀 있어서 수렴이 빠르거나, 데이터가 적어도 성능이 좋음
   - 현대에는 잘 않쓴다고 함
   
