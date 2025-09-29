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

`attention(Q, K, V) = softmax(QK^T / d  + mask)`

Q: Query vector

K: Key vector

V: Value vector

인코더/디코더 블록에 들어오는 토큰 벡터는 차원 d_model

멀티헤드 어텐션에서는 이걸 여러 “head”로 쪼갬.

h: head 개수
d_k = dimension of the key vectors

`d_k = d_v = d_model / h`


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
   
