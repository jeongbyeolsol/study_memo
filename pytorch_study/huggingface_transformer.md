# huggingface

## 함수/메서드 정리

### [`from_pretrained()`](https://huggingface.co/docs/transformers/v4.57.1/en/main_classes/model#transformers.PreTrainedModel.from_pretrained)

**사전 학습된 모델 불러오기**

모든 Hugging Face 객체(Config, Tokenizer, Model, Processor)가 공통으로 가지는 클래스 메서드

---

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


요약

| 클래스                | 역할             | 저장/불러오기 파일                      | 예시 클래스                                    |
| ------------------ | -------------- | ------------------------------- | ----------------------------------------- |
| `PretrainedConfig` | 모델의 구조/하이퍼파라미터 | `config.json`                   | `BertConfig`, `T5Config`                  |
| `PreTrainedModel`  | 실제 신경망 + 가중치   | `pytorch_model.bin`             | `BertModel`, `T5ForConditionalGeneration` |
| `Preprocessor`     | 입력 데이터 전처리기    | `tokenizer.json`, `vocab.txt` 등 | `AutoTokenizer`, `WhisperProcessor`       |

---

### [AutoClass](https://huggingface.co/docs/transformers/model_doc/auto)

다양한 모델 종류(BERT, GPT-2, T5, LLaMA 등)를 자동으로 감지하고 적절한 클래스로 로드해주는 팩토리(factory) 클래스

| AutoClass 이름                         | 로드하는 대상                       | 내부에서 로드되는 실제 클래스                                                         |
| ------------------------------------ | ----------------------------- | ------------------------------------------------------------------------ |
| `AutoTokenizer`                      | 토크나이저                         | `BertTokenizer`, `GPT2Tokenizer`, ...                                    |
| `AutoModel`                          | 기본 모델 (output: hidden states) | `BertModel`, `T5Model`, `GPT2Model`, ...                                 |
| `AutoModelForSequenceClassification` | 문장 분류 모델                      | `BertForSequenceClassification`, `RoBERTaForSequenceClassification`, ... |
| `AutoModelForCausalLM`               | 언어 생성 모델 (GPT류)               | `GPT2LMHeadModel`, `LlamaForCausalLM`, ...                               |
| `AutoModelForMaskedLM`               | MLM 모델 (BERT류)                | `BertForMaskedLM`, ...                                                   |
| `AutoProcessor`                      | 멀티모달 입력 처리기                   | `WhisperProcessor`, `CLIPProcessor`, ...                                 |

