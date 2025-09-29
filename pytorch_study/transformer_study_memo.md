# 인코더 구조

인코더를 직렬로 쌓은 구조 (원 논문 6개)

인코더 내부에 **self-attention**, **FFN**이 존재

👉 멀티헤드 어텐션의 head들은 병렬이지만,

👉 인코더 블록 자체는 직렬로 연결돼서 위로 쌓임.


# 디코더 구조

 **masked self-attention** -> **encoder-decoder-attention** -> **FFN**

- masked self-attention
  - 마스크 덕분에 각 토큰은 자신과 그 이전 토큰까지만 볼 수 있음

- **encoder-decoder-attention**
  - Query는 디코더의 은닉 상태, Key/Value는 인코더의 출력

# Attention

  attention(Q, K, V) = softmax(QK^T / d  + mask)


# 임베딩

텍스트(단어)를 숫자 벡터로 바꾸는 단계

1. 토큰화(tokenization): 문장을 단어/서브워드/문자 단위로 쪼갬.
2. 정수 인덱스 부여: 어휘(vocabulary) 딕셔너리를 만들어 각 토큰에 id를 붙임.
3. 임베딩 매트릭스 조회: 각 단어 id를 길이 d_model짜리 실수 벡터로 매핑

  embedding(x) = W * onehot(x)
  
원-핫 벡터에서 1이 있는 위치를 뽑으면, 사실상 **임베딩 매트릭스의 해당 행(row)**\을 꺼내는 것과 동일

