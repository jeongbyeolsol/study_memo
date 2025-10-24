# huggingface

## class 정리

*Each pretrained model inherits from three base classes*

### [`PretrainedConfig`](https://huggingface.co/docs/transformers/v4.57.1/en/main_classes/configuration#transformers.PretrainedConfig)

모델이 어떤 하이퍼파라미터를 가지고 만들어졌는지를 정의하는 객체

**모델 구조의 메타데이터(architecture)** 저장

모델 폴더 안에 config.json으로 저장


### [`PreTrainedModel`](https://huggingface.co/docs/transformers/v4.57.1/en/main_classes/model#transformers.PreTrainedModel)

모든 사전학습(pretrained) 모델이 상속받는 추상 기반 클래스

주요 역할

  - 모델의 가중치 저장/로드 (save_pretrained, from_pretrained)

  - Forward pass 정의 (즉, 입력→출력 계산)

  - 체크포인트에서 자동으로 불러오기

  - Trainer API에서 모델을 통합 형태로 사용하도록 표준화

중요 메서드

| 메서드                             | 설명                        |
| ------------------------------- | ------------------------- |
| `from_pretrained(name_or_path)` | 허깅페이스 허브 or 로컬에서 가중치 불러오기 |
| `save_pretrained(save_dir)`     | 모델 가중치 저장                 |
| `from_config(config)`           | Config 로부터 새 모델 인스턴스 생성   |
| `forward()`                     | 입력→출력 연산 정의               |


### `Preprocessor`

모델에 문자열, 이미지, 오디오 등을 넣기 전에 모델이 이해할 형태로 바꾸는 단계

주요 역할

  - 텍스트 → 토큰 ID (tokenizer.encode, __call__)

  - padding, truncation, special tokens 처리

  - 배치(batch) 입력 생성

  - 모델 출력 후 디코딩 지원 (tokenizer.decode)

---
