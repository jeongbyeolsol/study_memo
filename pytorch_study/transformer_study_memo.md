# 인코더 구조

인코더를 직렬로 쌓은 구조 (원 논문 6개)

인코더 내부에 **self-attention**, **FFN**이 존재

👉 멀티헤드 어텐션의 head들은 병렬이지만,

👉 인코더 블록 자체는 직렬로 연결돼서 위로 쌓임.


# 디코더 구조

 **self-attention** -> **encoder-decoder-attention** -> **FFN**
