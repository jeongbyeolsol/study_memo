# huggingface transformers

## 핵심 철학

1. 가능한 한 쉽고 빠르게 사용 가능하도록

2. 사전훈련된 모델(pretrained models)을 이용해 현실적이고 최첨단(“state-of-the-art”) 성능을 낼 수 있도록

핵심 클래스 단순화: 모델마다 너무 많은 사용자 대상 추상화(abstraction)를 두지 않고, 세 가지 표준 클래스만 배우면 대부분의 모델을 쓸 수 있도록 설계

- Configuration (모델 구조 및 하이퍼파라미터)

- Model (모델 자체)

- Preprocessor/Tokenizer/Processor (입력-출력 전처리) 

## 함수/메서드 정리

### [`from_pretrained()`](https://huggingface.co/docs/transformers/v4.57.1/en/main_classes/model#transformers.PreTrainedModel.from_pretrained)

**사전 학습된 모델 불러오기**

모든 Hugging Face 객체(Config, Tokenizer, Model, Processor)가 공통으로 가지는 클래스 메서드

- device_map="auto"는 모델 가중치를 가장 빠른 장치에 우선적으로 자동 할당

- dtype="auto"는 모델 가중치를 저장된 데이터 유형으로 직접 초기화하여 가중치를 두 번 로드하는 것을 방지(PyTorch는 기본적으로 torch.float32로 가중치를 로드)

| 핵심 기능   | 인자                                                    | 설명              |
| ------- | ----------------------------------------------------- | --------------- |
| 모델 지정   | `pretrained_model_name_or_path`                       | 모델 이름 또는 경로     |
| 메모리 최적화 | `low_cpu_mem_usage`, `device_map`, `torch_dtype`      | 대형 모델 로드 시 필수   |
| 양자화     | `quantization_config`, `load_in_8bit`, `load_in_4bit` | bitsandbytes 설정 |
| 보안/포맷   | `use_safetensors`, `trust_remote_code`                | 포맷/코드 로드 제어     |
| 기타      | `cache_dir`, `revision`, `offload_folder`             | 세부 제어용          |


### [`save_pretrained`](https://huggingface.co/docs/transformers/v4.57.1/en/main_classes/model#transformers.PreTrainedModel.save_pretrained)

모델 클래스안에 정의된, 모델 또는 토크나이저 객체를 로컬 디렉터리에 저장하는 메서드

| 인자                   | 설명                                             |
| -------------------- | ---------------------------------------------- |
| `save_directory`     | 저장할 디렉터리 경로                                    |
| `safe_serialization` | `True`일 경우 `.safetensors` 포맷으로 저장              |
| `state_dict`         | 특정 파라미터 dict만 저장하도록 지정                         |
| `push_to_hub`        | True면 Hugging Face Hub에 바로 업로드                 |
| `max_shard_size`     | 너무 큰 모델을 여러 shard로 나눠 저장할 때 크기 지정 (예: `"5GB"`) |


**“shard”**: 거대한 모델 가중치 파일을 여러 개로 쪼갠 “조각 파일(piece file)” 단위

### [`save_pretrained()`](https://huggingface.co/docs/transformers/v4.57.1/en/main_classes/model#transformers.PreTrainedModel.save_pretrained)

모델의 가중치, 설정(config), 그리고 추가 메타데이터를 로컬 폴더에 저장 → 나중에 from_pretrained()으로 그대로 복원할 수 있게 만들어주는 함수

| 인자                   | 설명                                  |
| -------------------- | ----------------------------------- |
| `save_directory`     | 저장할 폴더 경로                           |
| `safe_serialization` | `.safetensors` 포맷으로 저장 (보안 + 속도 향상) |
| `max_shard_size`     | 너무 큰 모델을 일정 단위로 나눠서 저장              |
| `state_dict`         | 수동으로 지정한 파라미터 dict만 저장할 때 사용        |
| `push_to_hub`        | Hugging Face Hub에 바로 업로드            |
| `is_main_process`    | 멀티GPU 환경에서 주 프로세스만 저장하도록 제어         |


---

### `tokenizer(...)`

`PreTrainedTokenizerBase` 클래스의 `__call__()` 메서드 오버라이드

내부적으로 encode_plus()를 호출, 문자열 → 토큰 ID / attention mask 딕셔너리 생성

return_tensors=: "pt" → PyTorch / "tf" → TensorFlow / "np" → NumPy

padding 옵션

| 값              | 설명                                  |
| -------------- | ----------------------------------- |
| `False` (기본값)  | 패딩 안 함 (문장마다 길이 다름)                 |
| `"longest"` / `True`   | 배치(batch) 내 **가장 긴 문장 길이**에 맞춰 패딩   |
| `"max_length"` | 지정한 `max_length` 길이에 맞춰 패딩          |

truncation 옵션

| 값                                | 설명                               |
| -------------------------------- | -------------------------------- |
| `False` (기본값)                    | 아무 것도 자르지 않음                     |
| `True` / `"longest_first"`     | `max_length` 기준으로 초과된 부분 잘라냄  / `max_length`이 주어지지 않을 경우, 모델에서 허용되는 최대 길이로 잘라냄  |
| `"only_first"` / `"only_second"` | 문장쌍 입력일 때 한쪽만 자름 (`max_length` 기준 초과 부분 잘라냄  / `max_length` 없는 경우 허용되는 최대 길이로 잘라냄  )                |

### `model.generate(...)`

`GenerationMixin` 클래스에서 정의된 공통 메서드

입력(input_ids, attention_mask)을 받아 토큰 예측 루프 수행

### `tokenizer.batch_decode()`

`PreTrainedTokenizerBase` 클래스의 메서드

모델이 생성한 토큰 ID 시퀀스를 문자열로 복원

---

### [`infer_device()`](https://huggingface.co/docs/transformers/v4.57.1/en/internal/file_utils#transformers.infer_device)

현재 사용 가능한 GPU/CPU 디바이스를 자동으로 감지해주는 함수

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
| [`AutoModel`](https://huggingface.co/docs/transformers/v4.57.1/en/model_doc/auto#transformers.AutoModel)                          | 기본 모델 (output: hidden states) | `BertModel`, `T5Model`, `GPT2Model`, ...                                 |
| `AutoModelForSequenceClassification` | 문장 분류 모델                      | `BertForSequenceClassification`, `RoBERTaForSequenceClassification`, ... |
| [`AutoModelForCausalLM`](https://huggingface.co/docs/transformers/v4.57.1/en/model_doc/auto#transformers.AutoModelForCausalLM)               | 언어 생성 모델 (GPT류)               | `GPT2LMHeadModel`, `LlamaForCausalLM`, ...                               |
| `AutoModelForMaskedLM`               | MLM 모델 (BERT류)                | `BertForMaskedLM`, ...                                                   |
| `AutoProcessor`                      | 멀티모달 입력 처리기                   | `WhisperProcessor`, `CLIPProcessor`, ...                                 |

  - “기본 모델”이란, 사전학습(pretrained)된 Transformer의 “언어 이해/표현(인코딩)” 부분만을 로드한 모델

---

### `BitsAndBytesConfig`

transformers 라이브러리에서 bitsandbytes 기반의 양자화 설정을 캡슐화한 클래스

세 가지 층위로 양자화의 세부 동작을 제어

  - 가중치 로딩 방식 (8bit / 4bit / full precision)
  - 계산 시 데이터 타입 (float16, bfloat16, float32 등)
  - 양자화 세부 방식 (NF4, FP4, double quantization 등)

주요 파라미터 정리

| 속성명                         | 타입          | 설명                    | 예시 / 추천                                 |
| --------------------------- | ----------- | --------------------- | --------------------------------------- |
| `load_in_8bit`              | bool        | 8비트 양자화 활성화           | `True`                                  |
| `load_in_4bit`              | bool        | 4비트 양자화 활성화           | `True`                                  |
| `bnb_4bit_quant_type`       | str         | 4비트 양자화 방식 선택         | `"nf4"` (Normalized Float 4) or `"fp4"` |
| `bnb_4bit_compute_dtype`    | torch.dtype | 연산 시 사용 dtype         | `torch.float16`, `torch.bfloat16`       |
| `bnb_4bit_use_double_quant` | bool        | 2단계 양자화 사용            | `True` (VRAM 절약↑)                       |
| `llm_int8_threshold`        | float       | 8bit 변환 시 예외처리 기준     | 기본 `6.0` (threshold↑ → 정확도↑, 메모리↓)      |
| `llm_int8_has_fp16_weight`  | bool        | 일부 가중치를 FP16으로 유지     | `True` (mixed precision)                |
| `llm_int8_skip_modules`     | list        | 양자화 제외할 모듈 이름         | `["lm_head"]` 등                         |
| `bnb_4bit_quant_storage`    | torch.dtype | 양자화된 값 저장 포맷          | `torch.uint8` (기본)                      |
| `tpu_vm_mode`               | bool        | TPU 환경용 (일반적으로 False) | `False`                                 |

---

### [`Trainer`](https://huggingface.co/docs/transformers/v4.57.1/en/main_classes/trainer#api-reference%20][%20transformers.Trainer)

**훈련 루프(Training Loop)**\의 고수준 클래스

주요 구성 요소
| 구성                               | 역할                             |
| -------------------------------- | ------------------------------ |
| `model`                          | 학습시킬 모델 객체 (`AutoModelFor*`)   |
| `args`                           | `TrainingArguments` (훈련 설정)    |
| `train_dataset` / `eval_dataset` | `datasets` 형태의 데이터셋            |
| `tokenizer`                      | (선택) 토크나이저, 자동 패딩/디코딩          |
| `compute_metrics`                | (선택) 평가 지표 함수 (accuracy, F1 등) |
| `data_collator`                  | (선택) 배치 구성 로직 (패딩, 마스크 등)      |



### [`TrainingArguments`](https://huggingface.co/docs/transformers/v4.57.1/en/main_classes/trainer#transformers.TrainingArguments)

Trainer: 실제 학습 진행 담당

TrainingArguments: 학습 설정 (batch size, epoch 수, 로그 주기 등)

주요 옵션

| 인자                            | 설명                                   | 예시                          |
| ----------------------------- | ------------------------------------ | --------------------------- |
| `output_dir`                  | 체크포인트 저장 폴더                          | `"./results"`               |
| `num_train_epochs`            | 학습 epoch 수                           | `3`                         |
| `per_device_train_batch_size` | 배치 크기                                | `8`                         |
| `learning_rate`               | 학습률                                  | `2e-5`                      |
| `weight_decay`                | 가중치 감쇠                               | `0.01`                      |
| `logging_dir`                 | TensorBoard 로그 저장 위치                 | `"./logs"`                  |
| `evaluation_strategy`         | 평가 시점 (`"no"`, `"steps"`, `"epoch"`) | `"epoch"`                   |
| `save_strategy`               | 저장 주기 (`"steps"`, `"epoch"`)         | `"epoch"`                   |
| `fp16`                        | half precision 훈련                    | `True`                      |
| `gradient_accumulation_steps` | 그래디언트 누적                             | `4`                         |
| `lr_scheduler_type`           | 학습률 스케줄러                             | `"linear"`, `"cosine"`, ... |

---

### [`pipeline`](https://huggingface.co/docs/transformers/main_classes/pipelines)

역할: 모델+토크나이저+전처리/후처리를 묶어 한 줄로 추론하게 해주는 고수준 래퍼.

자동화: 태스크에 맞는 기본 전처리/후처리를 붙이고, PyTorch/TF 자동 감지, GPU/CPU 디바이스 설정도 간편.

커스텀: 필요하면 직접 로드한 model, tokenizer, feature_extractor, processor를 끼워넣을 수 있음.

```python
from transformers import pipeline

pipe = pipeline(
    task,                 # "text-generation", "sentiment-analysis", "summarization", ...
    model=None,           # 모델 이름 또는 로컬 경로 (없으면 태스크의 기본 모델)
    tokenizer=None,       # 토크나이저 (생략 가능)
    device=None,          # 0 또는 "cuda:0" (GPU), -1 (CPU)
    framework=None,       # "pt" 또는 "tf" (대부분 자동 감지)
    return_tensors=False, # True면 모델 프레임워크의 텐서 반환
    **task_specific_kwargs
)
```

| 구분            | 매개변수                       | 설명                                                                                          |
| ------------- | -------------------------- | ------------------------------------------------------------------------------------------- |
| **태스크 선택** | `task`                     | 수행할 작업 (예: `"text-generation"`, `"summarization"`, `"translation"`, `"sentiment-analysis"`) |
| **모델 설정**  | `model`                    | 사용할 모델 이름 (ex. `"gpt2"`, `"bert-base-uncased"`)                                             |
|               | `tokenizer`                | 모델에 맞는 토크나이저 (대부분 자동 선택, 따로 지정할 일 적음)                                                       |
| **입력 처리**  | `padding`                  | 문장 길이를 맞춤 — `"max_length"` 또는 `"longest"`                                                   |
|               | `truncation`               | 긴 문장을 자름 — `True`                                                                           |
|               | `max_length`               | 입력 최대 길이 제한                                                                                 |
| **장치 선택**  | `device`                   | CPU(`-1`), GPU(`0`) 선택                                                                      |
| **생성 제어**  | `max_new_tokens`           | 생성 모델에서 새로 만들 토큰 수 제한                                                                       |
|               | `do_sample`, `temperature` | 텍스트 생성 다양성 조절 (`do_sample=True`, `temperature=0.7` 추천)                                      |
| **배치 처리**  | `batch_size`               | 여러 문장 처리 시 한 번에 넣을 크기                                                                       |


| 태스크                              | 추가 인자                                                                                                        | 설명                                               |
| -------------------------------- | ------------------------------------------------------------------------------------------------------------ | ------------------------------------------------ |
| **text-generation**              | `max_new_tokens`, `temperature`, `top_k`, `top_p`, `num_return_sequences`, `do_sample`, `repetition_penalty` | 텍스트 생성 제어                                        |
| **summarization**                | `max_length`, `min_length`, `truncation`, `num_beams`                                                        | 요약 길이 및 탐색 범위                                    |
| **translation**                  | `max_length`, `num_beams`, `forced_bos_token_id`                                                             | 번역 설정                                            |
| **question-answering**           | `top_k`, `max_answer_len`, `handle_impossible_answer`                                                        | 정답 후보 개수, 최대 길이                                  |
| **text-classification**          | `top_k`, `function_to_apply`, `aggregation_strategy`                                                         | 상위 라벨 수, 확률 변환 함수                                |
| **token-classification**         | `aggregation_strategy`                                                                                       | NER 등에서 엔티티 병합 방식 (`simple`, `first`, `average`) |
| **image-classification**         | `top_k`                                                                                                      | 상위 k개 라벨                                         |
| **automatic-speech-recognition** | `chunk_length_s`, `stride_length_s`, `return_timestamps`                                                     | 오디오 분할 및 타임스탬프 반환                                |


---

### `DataCollatorWithPadding`

배치(batch)를 만들 때, 각 문장의 길이를 확인하고 그중 가장 긴 문장에 맞춰 동적으로 패딩을 넣어줌

문장 전체가 아니라 배치 단위로 패딩 길이를 다르게 맞춰주는 객체

함수처럼 호출하는 형태로 사용


---

### `PreTrainedTokenizerBase`

Transformers 라이브러리의 모든 토크나이저(tokenizer)들의 기반(Base) 클래스

핵심 메서

1. `__call__()`: 문자열을 토큰화 + 인덱스화 + 패딩까지 한 번에 수행

2. `encode()` / `decode(ids)`: 문자열을 토큰 ID 리스트로 변환 / 토큰 ID 리스트를 다시 문자열로 복원

3. `batch_encode_plus()`: 문장 여러 개를 한 번에 처리하는 함수 (batch input용)

4. `save_pretrained()` / `from_pretrained()`: 모델 토크나이저를 저장/불러오기 위한 표준 메서드→ 모델과 동일한 경로 구조 유지 가능

5. `convert_tokens_to_ids()` / `convert_ids_to_tokens()`: 문자열 토큰 ↔ 정수 ID 변환

| 속성 이름                                                          | 설명                                                |
| -------------------------------------------------------------- | ------------------------------------------------- |
| `vocab_size`                                                   | 토크나이저 어휘 크기                                       |
| `pad_token_id`, `cls_token_id`, `sep_token_id`, `eos_token_id` | 특수 토큰 ID                                          |
| `padding_side`                                                 | `"left"` or `"right"`                             |
| `truncation_side`                                              | 문장 자를 때 방향                                        |
| `model_input_names`                                            | 모델이 요구하는 입력 이름 (예: `input_ids`, `attention_mask`) |
| `is_fast`                                                      | fast tokenizer인지 여부                               |

---

## 모듈 정리

### `configuration.py`

숨겨진 레이어 수, 어휘 크기, 활성화 함수 등과 같은 특정 속성을 정의

### `modeling.py`

각 레이어 내부에서 수행되는 레이어와 수학적 연산을 정의

`modeling.py` 파일은 `configuration.py`의 모델 속성을 받아 해당 모델을 구축
